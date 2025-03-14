
/*
* PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: parallel_3036030946.c
* NAME: Maposa Ngonidzashe
* UID:  3036030946
* Development Platform: workbench2
* Remark: I implemented everything:
A. Documentation
B. Report
C. Implementation
    1.(+2 points = 3 points) Achieved correct result & used multi-threading. Correct means generated text of multi-threading and sequential are identical with the same random seed.
    2.(+3 points = 6 points total) All in 1., and achieve >10% acceleration by multi-threading compared with sequential under 4 threads. Acceleration measurement is based on tok/s, acceleration must result from multi-threading instead of others like compiler (-O3), etc.
    3.(+2 points = 8 points total) All in 2., and reuse threads in multi-threading. Reuse threads means the number of threads created in the whole program must be constant as thr_count.
    4.(+3 points = 11 points total) All in 3., and mat_vec_mul and multi_head_attn use the same thread pool. Reusing the same thread pool means there’s only one pool and one thread group.
* How to compile separately: (gcc -o parallel parallel_3036030946.c -O2 -lm -lpthread)
*/

#include "common.h" // some common definitions
#include <unistd.h>       // for nearly everything :)
#include <stdio.h>        // for printf, sprintf, fgets
#include <stdlib.h>       // for malloc, calloc
#include <stdint.h>       // for uint8_t and uint64_t
#include <time.h>         // for time
#include <string.h>       // for memcpy and strcmp
#include <sys/resource.h> // for rusage collection
#include "model.h"// for Llama definitions -> no need to know
#include <math.h>
int pos = 0; // global position of generation
Transformer transformer; // transformer instance to be init
Tokenizer tokenizer;     // tokenizer instance to be init
Sampler sampler;         // sampler instance to be init

// YOUR CODE STARTS HERE
#include <pthread.h>
#include <semaphore.h> // uncomment this line if you use semaphore
// #include <stdbool.h>   // uncomment this line if you want true / false

// you may define global variables here 
//Global variables for everything
pthread_cond_t thread_cond   =  PTHREAD_COND_INITIALIZER;
pthread_mutex_t thread_mutex =  PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t completion_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t completion_mutex = PTHREAD_MUTEX_INITIALIZER;
int global_num_thr;               
pthread_t* threads;               
int* thread_ids;                 
struct rusage *usage_array;
int terminate = 0;
int global_threads_to_wake_up = 0;

//Global variables for MVM Computation
int global_rows_per_thread=0;
int global_rows_for_last_thread=0;
float *global_mvm_out;          
QuantizedTensor *global_vec;   
QuantizedTensor *global_mat;    
int global_col;                 
int global_row;                 
int *mvm_computation;
int thread_finished_count = 0;

//Global variables for MHA Computation
int global_heads_per_thread=0;
int global_heads_for_last_thread=0;
int *mha_computation;  
float *global_mha_out;     // Output tensor [head, head_size]
float *global_q;           // Query tensor [head, head_size]
float *global_key_cache;   // Cache of history key tensor [kv_head, seq_len, head_size]
float *global_value_cache; // Cache of history value tensor [kv_head, seq_len, head_size]
float *global_att;         // Buffer for attention score [head, seq_len]
int global_seq_len;        // Current sequence length
int global_n_heads;        // Number of heads
int global_head_size;      // Size of each head
int global_kv_dim;         // Key-Value dimension
int global_kv_mul;         // Key-Value multiplier



// function executed by each thread to complete mat_vec_mul
// @note: please modify the signature to what you want
void mat_vec_mul_task_func(int id) {
    int num_rows = 0;            // Rows to process
    int start_row = 0;          // Starting row index
    int end_row = 0;            // Ending row index

    if (id == (global_threads_to_wake_up - 1)){   // Last thread gets remaining rows
         num_rows = global_rows_for_last_thread;
    } else {
        num_rows = global_rows_per_thread;       // Other threads get standard rows
    }

    start_row = id * global_rows_per_thread;  // Calculate start index
    end_row = start_row + num_rows;           // Calculate end index

    for (int i = start_row; i < end_row; i++) {
        float val = 0.0f; // final value
        int32_t ival = 0; // integer value to be dequantized
        int in = i * global_col;   // 

        // for each column
        // GS is group size of quantization, not included in assignment
        // @note please don't parallel this loop
        for (int j = 0; j <= global_col - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t) global_vec->q[j + k]) * ((int32_t) global_mat->q[in + j + k]);
            }
            val += ((float) ival) * global_mat->s[(in + j) / GS] * global_vec->s[j / GS];
            ival = 0;
        }
        global_mvm_out[i] = val;
    }
}

// function executed by each thread to complete multi_head_attn
// @note: please modify the signature to what you want
void multi_head_attn_task_func(int id) {
    int num_heads = 0;            // Heads to process
    int start_head = 0;          // Starting head index
    int end_head = 0;            // Ending head index


    if (id == (global_threads_to_wake_up - 1)){     // Last thread gets remaining heads
         num_heads = global_heads_for_last_thread;
    }
    else{
        num_heads = global_heads_per_thread;       // Other threads get standard heads
    }
    start_head = id * global_heads_per_thread; // Calculate starting head index
    end_head = start_head + num_heads;          // Calculate ending head index

    for (int h = start_head; h < end_head; h++) {
    // Get the query vector for this head
    float* head_q = global_q + h * global_head_size;
    // Attention scores for this head
    float* head_att = global_att + h * global_seq_len;

    // Iterate over all timesteps, including the current one
    for (int t = 0; t <= pos; t++) {
        // Get the key vector for this head and at this timestep
        float* head_k = global_key_cache + t * global_kv_dim + (h / global_kv_mul) * global_head_size;

        // Calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < global_head_size; i++) {
            score += head_q[i] * head_k[i];
        }
        score /= sqrtf(global_head_size);

        // Save the score to the attention buffer
        head_att[t] = score;
    }

    // Softmax the scores to get attention weights, from 0..pos inclusively
    softmax(head_att, pos + 1);

    // Weighted sum of the values, store back into out
    float* head_out = global_mha_out + h * global_head_size;
    memset(head_out, 0, global_head_size * sizeof(float));
    for (int t = 0; t <= pos; t++) {
        // Get the value vector for this head and at this timestep
        float* head_v = global_value_cache + t * global_kv_dim + (h / global_kv_mul) * global_head_size;

        // Get the attention weight for this timestep
        float a = head_att[t];

        // Accumulate the weighted value into head out
        for (int i = 0; i < global_head_size; i++) {
            head_out[i] += a * head_v[i];
        }
    }
}

}


// thread function used in pthread_create
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void *thr_func(void *arg) {
    int thread_id = *(int*)arg; // Cast and dereference the argument
    int has_mvm_work;
    int has_mha_work;
    while (1){

        // Wait for work or termination signal; check if there's work to do
        pthread_mutex_lock(&thread_mutex);
        while (!mvm_computation[thread_id] && !mha_computation[thread_id] && !terminate) {
            pthread_cond_wait(&thread_cond, &thread_mutex);
        }  

        // Check for termination; if so, unlock and exit the thread
        if (terminate){
            pthread_mutex_unlock(&thread_mutex);
            getrusage(RUSAGE_THREAD, &usage_array[thread_id]);
            break;              
        }
        else {

            // Check for available work and unlock the mutex
            has_mvm_work = mvm_computation[thread_id];
            has_mha_work = mha_computation[thread_id];
            pthread_mutex_unlock(&thread_mutex);

            // Perform computations based on available work
            if (has_mvm_work){ 

                    // Execute MVM function
                    mat_vec_mul_task_func(thread_id);

                    // Mark MVM work as complete
                    pthread_mutex_lock(&thread_mutex);
                    mvm_computation[thread_id] = 0; 
                    pthread_mutex_unlock(&thread_mutex); 

                    // Update completion status by incrementing finished thread count and signal completion to main thread
                    pthread_mutex_lock(&completion_mutex);
                    thread_finished_count++;
                    pthread_cond_signal(&completion_cond);
                    pthread_mutex_unlock(&completion_mutex);

                }

            else if (has_mha_work){
                    // Execute MHA function
                    multi_head_attn_task_func(thread_id);

                    // Mark MHA work as complete
                    pthread_mutex_lock(&thread_mutex);
                    mha_computation[thread_id] = 0; 
                    pthread_mutex_unlock(&thread_mutex);

                    // Update completion status by incrementing finished thread count and signal completion to main thread
                    pthread_mutex_lock(&completion_mutex);
                    thread_finished_count++;
                    pthread_cond_signal(&completion_cond);
                    pthread_mutex_unlock(&completion_mutex);
                }

            }  
        }    
    // Exit the thread
    pthread_exit(NULL);              
    }


// function to initialize thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void init_thr_pool(int num_thr) {
    global_num_thr = num_thr; // Set the global number of threads
    thread_ids = malloc(num_thr * sizeof(int)); // Allocate memory for thread IDs
    threads = malloc(num_thr * sizeof(pthread_t)); // Allocate memory for thread handles
    mvm_computation = malloc(num_thr * sizeof(int)); // Allocate memory for MVM work flags
    mha_computation = malloc(num_thr * sizeof(int)); // Allocate memory for MHA work flags
    usage_array = malloc(num_thr * sizeof(struct rusage)); // Allocate memory for resource usage tracking

    // Initialize the computation arrays for each thread
    pthread_mutex_lock(&thread_mutex);
    for (int i = 0; i<num_thr; i++){
        mvm_computation[i] = 0;
        mha_computation[i] = 0;
        }
    pthread_mutex_unlock(&thread_mutex);

    // Create the threads and assign their IDs
    for (int i = 0; i<num_thr; i++){
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, thr_func, (void*)&thread_ids[i]);
    }
}

// function to close thread pool
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void close_thr_pool() {
    terminate = 1; // Signal all threads to terminate
    pthread_cond_broadcast(&thread_cond); // Wake up all waiting threads

    // Wait for all threads to complete
    for (int i = 0; i < global_num_thr; i++) {
        pthread_join(threads[i], NULL);
    }

    // Print resource usage for each thread
     for (int i = 0; i < global_num_thr; i++) {
        printf("\033[0;32mThread %d has completed - user: %.4f s, system: %.4f s \033[0m\n",
            i, 
            (usage_array[i].ru_utime.tv_sec + usage_array[i].ru_utime.tv_usec/1000000.0), 
            (usage_array[i].ru_stime.tv_sec + usage_array[i].ru_stime.tv_usec/1000000.0)); 
     }
     
    // Get and print resource usage for the main thread
    struct rusage main_usage;
    getrusage(RUSAGE_THREAD, &main_usage); // to avoid child threads
    printf("\033[0;32mmain thread - user: %.4f s, system: %.4f s \033[0m\n",
    (main_usage.ru_utime.tv_sec + main_usage.ru_utime.tv_usec/1000000.0),
    (main_usage.ru_stime.tv_sec + main_usage.ru_stime.tv_usec/1000000.0));

    // Get and print total resource usage for the whole process
    struct rusage whole_usage;
    getrusage(RUSAGE_SELF, &whole_usage);
    printf("\033[0;32mWhole process - user: %.4f s, system: %.4f s \033[0m\n",
    (whole_usage.ru_utime.tv_sec + whole_usage.ru_utime.tv_usec/1000000.0),
    (whole_usage.ru_stime.tv_sec + whole_usage.ru_stime.tv_usec/1000000.0));

    // Clean up: destroy condition variables and mutexes
    pthread_cond_destroy(&thread_cond);
    pthread_mutex_destroy(&thread_mutex);
    pthread_cond_destroy(&completion_cond);
    pthread_mutex_destroy(&completion_mutex);

    // Free allocated resources
    free(threads); // Free the array of thread handles
    free(thread_ids); // Free the array of thread IDs
    free(mvm_computation); // Free the MVM computation flags array
    free(mha_computation); // Free the MHA computation flags array
    free(usage_array); // Free the usage array


    
}

// ----------------------------------------------------------------------------
// entry function for multi-threading matrix multiplication
// @note: YOU CAN NOT MODIFY this FUNCTION SIGNATURE!!!
void mat_vec_mul(float* out, QuantizedTensor *vec, QuantizedTensor *mat, int col, int row) {
    // Set global values for MVM Computation for all worker threads to access
    global_mvm_out = out;
    global_vec = vec;
    global_mat = mat;
    global_col = col;
    global_row = row;

    // Calculate the number of rows each thread will process
    int rows_per_thread = ceil((float)row / global_num_thr);
    int threads_to_wake_up = ceil((float)row / rows_per_thread);
    int rows_for_last_thread = row - ((threads_to_wake_up - 1) * rows_per_thread) ;


    // Reset the finished thread count
    pthread_mutex_lock(&completion_mutex);
    thread_finished_count = 0;
    pthread_mutex_unlock(&completion_mutex);


    // Activate MVM computation for the relevant threads
    pthread_mutex_lock(&thread_mutex);
    for (int i = 0; i < threads_to_wake_up; i++) {
        mvm_computation[i] = 1;                           //Mark threads to perform computation
    }
    global_rows_per_thread = rows_per_thread;             // Set global rows per thread
    global_rows_for_last_thread = rows_for_last_thread;   // Set rows for the last thread
    global_threads_to_wake_up = threads_to_wake_up;       // Set the number of threads to wake up
    pthread_mutex_unlock(&thread_mutex);

    // Notify all waiting threads to start processing
    pthread_cond_broadcast(&thread_cond);


    // Wait for all activated threads to complete their work
    pthread_mutex_lock(&completion_mutex);
    while (thread_finished_count < threads_to_wake_up) {
        pthread_cond_wait(&completion_cond, &completion_mutex);
    }

    pthread_mutex_unlock(&completion_mutex);

//     //Turn off mvm computation 
//    pthread_mutex_lock(&thread_mutex);
//     for (int i = 0; i < threads_to_wake_up; i++) {
//         mvm_computation[i] = 0; 
//     }
//     pthread_mutex_unlock(&thread_mutex);   
}

// ----------------------------------------------------------------------------
// entry function for multi-threading multi-head-attention
// @note: YOU CAN NOT MODIFY FUNCTION SIGNATURE!!!
void multi_head_attn(
    float* out,         // output tensor [head, head_size]
    float* q,           // query tensor  [head, head_size]
    float* key_cache,   // cache of history key tensor   [kv_head, seq_len, head_size]
    float* value_cache, // cache of history value tensor [kv_head, seq_len, head_size]
    float* att,         // buffer for attention score [head, seq_len]
    int seq_len,
    int n_heads,
    int head_size,
    int kv_dim,
    int kv_mul) {

    // Set global variables for multi-head attention computation
    global_mha_out = out;
    global_q = q;
    global_key_cache = key_cache;
    global_value_cache = value_cache;
    global_att = att;
    global_seq_len = seq_len;
    global_n_heads = n_heads;
    global_head_size = head_size;
    global_kv_dim = kv_dim;
    global_kv_mul = kv_mul;

    int heads_per_thread = ceil((float)n_heads / global_num_thr);               // Calculate number of heads each thread will process
    int threads_to_wake_up = ceil((float)n_heads / heads_per_thread);           // Determine threads to activate
    int heads_for_last_thread = n_heads - ((threads_to_wake_up - 1) * heads_per_thread) ; // Heads for last thread

    // Reset the finished thread count
    pthread_mutex_lock(&completion_mutex);
    thread_finished_count = 0;
    pthread_mutex_unlock(&completion_mutex);

    // Activate MHA computation for the relevant threads
    pthread_mutex_lock(&thread_mutex);
    for (int i = 0; i < threads_to_wake_up; i++) {
        mha_computation[i] = 1; 
    }
    global_heads_per_thread = heads_per_thread; // Set global heads per thread
    global_heads_for_last_thread = heads_for_last_thread; // Set heads for last thread
    global_threads_to_wake_up = threads_to_wake_up; // Set number of threads to wake up
    pthread_mutex_unlock(&thread_mutex);

    // Notify all waiting threads to start processing
    pthread_cond_broadcast(&thread_cond);

    // Wait for all activated threads to complete their work
    pthread_mutex_lock(&completion_mutex);
    while (thread_finished_count < threads_to_wake_up) {
        pthread_cond_wait(&completion_cond, &completion_mutex);
    }
    pthread_mutex_unlock(&completion_mutex);

    // //Turn off mha computation
    // pthread_mutex_lock(&thread_mutex);
    // for (int i = 0; i < threads_to_wake_up; i++) {
    //     mha_computation[i] = 0; 
    // }
    // pthread_mutex_unlock(&thread_mutex);   

}

// YOUR CODE ENDS HERE

// ----------------------------------------------------------------------------
// forward Transformer, you're not allowed to modify this part
float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);

        mat_vec_mul(s->q, &s->xq, w->wq + l, dim, dim);
        mat_vec_mul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        mat_vec_mul(s->v, &s->xq, w->wv + l, dim, kv_dim);


        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));
        // printf("I AM HERE!!!");

        multi_head_attn(s->xb, s->q, s->key_cache + loff, s->value_cache + loff, s->att, 
            p->seq_len, p->n_heads, head_size, kv_dim, kv_mul);

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(&s->xq, s->xb, dim);
        mat_vec_mul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        mat_vec_mul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        mat_vec_mul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    quantize(&s->xq, x, dim);

    mat_vec_mul(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// generation loop, you're not allowed to modify this part
void generate(char *prompt) {
    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+6) * sizeof(int)); // +6 reserved for prompt template
    encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    int next;        // place holder for next token
    int token = prompt_tokens[0]; // place holder of prev token, kickoff as prompt_tokens[0]
    int end_pos = pos + MAX_NEW_TOKENS + num_prompt_tokens;
    int start_pos = pos;
    long start_time = 0; // to be lazy iniialzied
    while (pos < end_pos) {

        // forward the transformer to get logits for the next token

        float* logits = forward(&transformer, token, pos);

        if (pos < start_pos + num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos - start_pos + 1];
        } else if (pos == end_pos - 2) {
            // reaching the end, force it to close by <|im_end|>
            next = 2; // := <|im_end|>
        } else {
            // otherwise sample the next token from the logits
            next = sample(&sampler, logits);
        }

        pos++;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(&tokenizer, token, next);
        if (pos >= num_prompt_tokens) {
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }

        token = next;

        // init the timer here because the first iteration can be slower
        if (start_time == 0) { start_time = time_in_ms(); }
    }
    printf("\n");

    long end_time = time_in_ms();
    // \033[0;32m set color to green and \033[0m reset to default, they won't be generate by LLM
    fprintf(stdout, "\033[0;32mlength: %d, speed (tok/s): %.4f \033[0m\n", 
        pos, (pos - start_pos) / (float) (end_time - start_time) * 1000);
    
    free(prompt_tokens);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *model_path     = "model.bin";  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature    = 0.6f;  // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp           = 0.9f;  // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    char *prompt         = NULL;  // prompt strings
    int num_prompt       = 0; // number of prompts
    uint64_t rng_seed    = 0; // seed rng with time by default
    int num_thr          = 0;

    if (argc == 4) {
        num_thr  = atoi(argv[1]);
        rng_seed = atoi(argv[2]);
        prompt   = argv[3];
    } else {
        fprintf(stderr, "Usage:   ./seq <num_thr> <seed> <prompt>\n");
        fprintf(stderr, "Example: ./seq 4 42 \"What is Fibonacci Number?\"\n");
        fprintf(stderr, "Note:    <prompt> must be quoted with \"\", only one prompt supported\n");
        exit(1);
    }

    // parameter validation/overrides
    if (num_thr <= 0 || num_thr > 16) { 
        fprintf(stderr, "num_thr must between 1 and 16 \n");
        exit(EXIT_FAILURE);
    }
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    build_transformer(&transformer, model_path);
    // build the Tokenizer via the tokenizer .bin file
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);
    // build the Sampler
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // initialize thread pool
    init_thr_pool(num_thr);

    printf("user: %s \n", prompt);
    // perform multi-threading generation
    generate(prompt);

    // close thread pool
    close_thr_pool();

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}