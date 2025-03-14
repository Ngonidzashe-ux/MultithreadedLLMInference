/*
* PLEASE WRITE DOWN FOLLOWING INFO BEFORE SUBMISSION
* FILE NAME: main_3036030946.c
* NAME: Maposa Ngonidzashe
* UID:  3036030946
* Development Platform: workbench2
* Remark: (How much you implemented?)
Implemented Everything:
1. [1pt] Build a chatbot that accept user input, inference and print generated text to stdout.
2. [2pt] Separate Inference Process and Main Process (for chatbot interface) via pipe and exec
3. [1pt] Correctly forward user input from main process to subprocess via pip
4. [1pt] Correctly synchronize the main process with the inference process for the completion of
inference generation.
5. [2pt] Correctly handle SIGINT that terminates both main and inference processes and collect
the exit status of the inference process.
6. [2.5pt] Correctly parse the /proc file system of the inference process during inferencing to
collect and print required fields to stderr.
7. [0.5pt] Correctly calculate the cpu usage in percentage and print to stderr.
8. [2pt] Correctly use SYS_sched_setattr to set the scheduling policy and parameters
I also included documentation adn the Report
* How to compile separately: (gcc -o main main_3036030946.c)
*/

#include "common.h"  // common definitions
#include <stdio.h>   // for printf, fgets, scanf, perror
#include <stdlib.h>  // for exit() related
#include <unistd.h>  // for folk, exec...
#include <wait.h>    // for waitpid
#include <signal.h>  // for signal handlers and kill
#include <string.h>  // for string related 
#include <sched.h>   // for sched-related
#include <syscall.h> // for syscall interface

#define READ_END       0    // helper macro to make pipe end clear
#define WRITE_END      1    // helper macro to make pipe end clear
#define SYSCALL_FLAG   0    // flags used in syscall, set it to default 0

// Define Global Variable, Additional Header, and Functions Here
pid_t child_pid;
int status; 
volatile sig_atomic_t sigusr1_received = 0;


//Handler for SIGINT signal.
void sigint_handler(int signum){
	kill(child_pid, SIGINT); //Send SIGINT signal to child
	
}

//Handler for SIGUSR1 signal.
void sigusr1_handler(int signum) {
    sigusr1_received = 1;  //Change to 1 when SIGUSR1 is received
}

//Handler for SIGUSR2 signal.
void sigusr2_handler(int signum){
	int status;  //Initialise status variable
	waitpid(child_pid,&status, 0); //Call waitpid() to wait for the child and collect exit status
	printf("\nChild exited with %d\n", WEXITSTATUS(status)); //Print out status code
	fflush(stdout); //Immediate print out of the output buffer contents
	exit(0); //Exit
}

//Function for reading the /proc/<pid>/status file
void read_proc_stat(int pid, int *extracted_pid, char *tcomm, char *state, unsigned long *utime, unsigned long *stime, unsigned long *vsize, int *task_cpu, int *nice, int *policy) {
    char proc_path[256]; //Initialised character array for storing the proc path
    snprintf(proc_path, sizeof(proc_path), "/proc/%d/stat", pid);

    FILE *file = fopen(proc_path, "r"); //Opening file in read model
    if (!file) {
        perror("/PROC file cant be opened");
        exit(EXIT_FAILURE);
    }
    char statistics[1024]; //Initialise a char array called statistics for storing the information
    fgets(statistics, sizeof(statistics), file); //Read until new line.
    fclose(file);//Close file after reading

    // Extract the required fields using sscanf and using %*s to ignore unnecessary ones.
    sscanf(statistics, "%d %s %c %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %lu %lu %*s %*s %*s %d %*s %*s %*s %lu %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %d %*s %d",
           extracted_pid, tcomm, state, utime, stime, nice, vsize, task_cpu, policy );
}


// Function to set the scheduling policy and nice value
int set_scheduling_policy(pid_t pid, unsigned int POLICY, int nice_value) {
    struct sched_attr attr;

    // Initialize the sched_attr structure
    attr.size = sizeof(attr);
    attr.sched_policy = POLICY;      // Set the scheduling policy
    attr.sched_flags = 0;            // internal use only
    attr.sched_nice = nice_value;    // Set the nice value
    attr.sched_priority = 0;         // NOT USED FOR NORMAL SCHEDULING
    attr.sched_runtime = 0;          // NOT USED FOR NORMAL SCHEDULING
    attr.sched_deadline = 0;         // NOT USED FOR NORMAL SCHEDULING
    attr.sched_period = 0;           // NOT USED FOR NORMAL SCHEDULING
    attr.sched_util_min = 0;         // NOT USED
    attr.sched_util_max = 0;         //NOT USED

    // Making a syscall to set the scheduling attributes
    if (syscall(SYS_sched_setattr, pid, &attr, 0) < 0) {
        HANDLE_ERROR(); 
        return -1;
    }
    return 0;
}

//The main function
int main(int argc, char *argv[]) {
    char* seed; 
    if (argc == 2) {
        seed = argv[1];
    } else if (argc == 1) {
        // use 42, the answer to life the universe and everything, as default
        seed = "42";
    } else {
        fprintf(stderr, "Usage: ./main <seed>\n");
        fprintf(stderr, "Note:  default seed is 42\n");
        exit(1);
    }

	// Write your main logic here

	//SIGNAL HANDLERS
	signal(SIGINT, sigint_handler); 	//Direct SIGINT signal to the SIGINT handler
	signal(SIGUSR1, sigusr1_handler);   //Direct SIGUSR1 signal to the SIGUSR1 handler
	signal(SIGUSR2, sigusr2_handler);	//Direct SIGUSR2 signal to the SIGUSR2 handler

	//SCHEDULING VARIABLES
 	pid_t pid = getpid(); // Get current process ID
    unsigned int POLICY; // Variable to hold scheduling policy
    int nice_value;      // Variable to hold nice value

	//MONITORING VARIABLES
	int extracted_pid;
    char tcomm[256];
    char state;
    unsigned long  vsize;
    int task_cpu, nice, policy;
	unsigned long prev_utime, prev_stime;
    unsigned long curr_utime, curr_stime;

    // Set the scheduling policy and nice value for the current process
    POLICY = SCHED_OTHER;  // SCHED_OTHER or SCHED_BATCH or SCHED_IDLE 
    nice_value = 0;        // 0, 2, 10
    if (set_scheduling_policy(pid, POLICY, nice_value) < 0) {
        fprintf(stderr, "Failed to set scheduling policy\n");
        return EXIT_FAILURE;
    }

	//Execution Logic

	int pfd[2];
	pipe(pfd); //Set up pipe for communication with child
	child_pid = fork(); //Create child

	if (child_pid < 0){ //Error handling during fork
		printf("There is an error");
    	fflush(stdout);
	}  

	else if (child_pid == 0) { //CHILD
		close(pfd[WRITE_END]); //Close writing end of pipe
		dup2(pfd[READ_END], STDIN_FILENO); //Redirect the read end to the stdin of child
        execlp("./inference", "./inference", seed, NULL); //Replace child process with the "./inference" executable.
	}

	else {  //PARENT
		close(pfd[READ_END]);	//Close reading end
		char prompt[MAX_PROMPT_LEN]; //Initialise a character array called prompt that holds a maximum length of MAX_PROMPT_LEN.
		
		for (int i=0; i<4; i++){
			printf(">>> "); 
			fgets(prompt, sizeof(prompt), stdin);	//Read user input
			write(pfd[WRITE_END], prompt, (strlen(prompt)));	//Write user input to the pipe

            // Initialise previous user and system to zero
			prev_utime = 0;
			prev_stime = 0;
 
			// Continously run the loop to read from the /proc/status
			while (1) {
				read_proc_stat(child_pid, &extracted_pid, tcomm, &state, &curr_utime, &curr_stime, &vsize, &task_cpu, &nice, &policy); //pass in the variables to store the statistics

			// Calculate CPU usage percentage based on utime and stime
			unsigned long utime_diff = curr_utime - prev_utime;
			unsigned long stime_diff = curr_stime - prev_stime;
			double total_time = (utime_diff + stime_diff) ;
			double cpu_percentage = ((total_time / 30.0) * 100);

			//Run if sigusr1 is received from child
			if (sigusr1_received) { 
					sigusr1_received = 0; //Reset the sigusr1_received variable
					break;  // Exit the loop 
				}
        	// Otherwise Print the required fields to stderr
        	fprintf(stderr, "[pid] %d [tcomm] %s [state] %c [policy] %s [nice] %d [vsize] %lu [task_cpu] %d [utime] %lu [stime] %lu [cpu%%] %.2f%%\n",
                extracted_pid, tcomm,  state, get_sched_name(policy), nice, vsize, task_cpu, curr_utime, curr_stime, cpu_percentage);
        	fflush(stderr);

        	// Update previous values for the next iteration
			prev_utime = curr_utime;
			prev_stime = curr_stime;  

			usleep(300000);  // 300ms = 300,000µs  
            }
        }

		//Wait for the child to terminate to collect exit status and print exit status on new line.
		waitpid(child_pid, &status, 0);
		printf("Child exited with %d\n", WEXITSTATUS(status));
	}
    return EXIT_SUCCESS;
}
