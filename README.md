# MultithreadedLLMInference

**MultithreadedLLMInference** is a project developed for **COMP3230 Principles of Operating Systems** at The University of Hong Kong. It consists of two programming assignments leveraging the **Llama3 (Smoll.M) large language model (LLM)**:

1. **PA1 - Chatbot Interface**: A Unix-based chatbot interface using **process separation, pipes, signals, and the `/proc` filesystem** to interact with an LLM inference engine, supporting configurable scheduling policies.
2. **PA2 - Multithreaded Inference**: A multithreaded LLM inference engine using **POSIX Pthreads** to accelerate **matrix-vector multiplication and multi-head attention**, optimized with a reusable **thread pool**.

This project demonstrates **system programming, process management, and multithreading on Linux**, aligning with **Intended Learning Outcome 4 (Practicability)** of COMP3230.

## 📁 Project Structure
```plaintext
MultithreadedLLMInference/
├── Implement LLM ChatBot Interface/  # PA1: Chatbot Interface
│   ├── common.h                      # Shared macros (provided)
│   ├── main_[UID].c                  # Main process for user I/O
│   ├── inference_[UID].c             # Inference child process
│   ├── Makefile                      # Build script (UID updated)
│   ├── model.h                       # LLM model defs (provided)
│   └── avg_cpu_use.py                # CPU usage parser (provided)
├── Accelerate LLM Using MultiThreading/ # PA2: Multithreaded Inference
│   ├── common.h                      # Shared macros (provided)
│   ├── seq.c                         # Single-thread baseline (provided)
│   ├── parallel_[UID].c              # Multithreaded implementation
│   ├── Makefile                      # Build script (UID updated)
│   └── model.h                       # LLM model defs (provided)
├── .gitignore                        # Ignores binaries, logs
└── README.md                         # Project documentation
```

## ⚠️ Model Files
Large files (`model.bin`, `tokenizer.bin`, ~136MB each) exceed GitHub’s 100MB limit and are hosted externally. See [Model Files](#model-files) for details.

---
## 🏗 Assignment Details

### PA1: Chatbot Interface
**Objective**: Build a chatbot interface for **Llama3 (Smoll.M)** using **Unix process APIs**, separating user interaction (main process) from inference (child process).

**Key Features**:
- **Process Separation**: Uses `fork()` and `exec()` to isolate inference in a child process, with pipes for communication.
- **Signal Synchronization**: Implements **SIGUSR1, SIGINT** for inter-process coordination and safe termination.
- **Process Monitoring**: Reads `/proc/[pid]/stat` every 300ms to log **CPU usage, PID, and process state** to `stderr`.
- **Custom Scheduling**: Supports **SCHED_OTHER, SCHED_BATCH** via `SYS_sched_setattr` syscall.

**Build & Run:**
```bash
cd "Implement LLM ChatBot Interface"
make -B
./main <seed> 2>proc.log  # Example: ./main 42 2>proc.log
```

---
### PA2: Multithreaded Inference
**Objective**: Parallelize **Llama3 (Smoll.M) inference** using **POSIX Pthreads** to accelerate **matrix-vector multiplication and multi-head attention**.

**Key Features**:
- **Parallel Processing**: Distributes matrix rows and attention heads across threads.
- **Thread Pool**: Reuses a **fixed thread pool** for efficiency, reducing thread creation overhead.
- **Performance Boost**: Achieves **>10% increase in tokens/sec (tok/s)** vs. sequential baseline with **4 threads**.
- **System Usage Reporting**: Logs **per-thread and overall CPU usage** using `getrusage()`.

**Build & Run:**
```bash
cd "Accelerate LLM Using MultiThreading"
make -B
./parallel 4 42 "What is Fibonacci Number?"
```

---
## 🛠 Prerequisites
- **OS**: Linux (developed on macOS, tested on HKU's workbench2.cs.hku.hk)
- **Compiler**: GCC (e.g., `gcc -v` shows 13.2.0)
- **Libraries**: Standard C (`glibc`), `pthread.h`, `semaphore.h`, `-lm`
- **Model Files**: See [Model Files](#model-files)

---
## 🚀 Setup Instructions
### Clone the Repository
```bash
git clone https://github.com/Ngonidzashe-ux/MultithreadedLLMInference.git
cd MultithreadedLLMInference
```
### Update UID
Rename source files:
```plaintext
main_[UID].c → main_3031234567.c
inference_[UID].c → inference_3031234567.c
parallel_[UID].c → parallel_3031234567.c
```
Modify `Makefile` (replace `[UID]` on line 5 with your HKU UID).

### Download Model Files
Due to GitHub’s **100MB limit**, download externally:
- **[model.bin](#)** (136.40MB)
- **[tokenizer.bin](#)**

Place in both subdirectories or run:
```bash
cd "Implement LLM ChatBot Interface" && make prepare
cd "../Accelerate LLM Using MultiThreading" && make prepare
```

---
## 🏃 Usage
### PA1: Chatbot Interface
```bash
cd "Implement LLM ChatBot Interface"
./main 42 2>proc.log
```
```plaintext
>> What is Fibonacci Number?
[Generated text...]
>> Write a Python program to generate Fibonacci Numbers.
[Generated code...]
^C  # Ctrl+C to exit, displays child exit status
```
### PA2: Multithreaded Inference
```bash
cd "Accelerate LLM Using MultiThreading"
./parallel 4 42 "What is Fibonacci Number?"
```
```plaintext
# Outputs generated text, speed (tok/s), and thread usage
```

---
## 🔥 Challenges Overcome
- **PA1**: Synchronized processes with **pipes & signals**, parsed `/proc`, and implemented raw `SYS_sched_setattr`.
- **PA2**: Parallelized **matrix-vector multiplication & multi-head attention**, optimizing with a single **thread pool**.

## 🔮 Future Enhancements
- **PA1**: Add real-time **CPU usage visualization** in the shell.
- **PA2**: Experiment with **dynamic thread allocation** based on workload.

---
## 📜 References
- **[Llama2.c](https://github.com/karpathy/llama2.c) by Andrei Karpathy**
- **[Smoll.M](https://huggingface.co/) by Huggingface**
- **Linux `/proc` Filesystem & Pthreads Documentation**

## 📝 License
MIT License — Free to use and modify. *(Consider adding a LICENSE file.)*

## 👤 About the Author
Developed by **Ngonidzashe Maposa**, a Computer Engineering student at **The University of Hong Kong**. Passionate about **systems programming, AI, and optimization**. 

🔗 **Connect with me:** [LinkedIn](#) | [GitHub](https://github.com/Ngonidzashe-ux)
