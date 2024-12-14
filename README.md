# OS Scheduling Algorithms

Welcome to the **OS Scheduling Algorithms** repository! This project contains implementations of several popular CPU scheduling algorithms, including:

- **HRRN (Highest Response Ratio Next)**
- **SRT (Shortest Remaining Time)**
- **Round Robin**
- **SJF (Shortest Job First)**
- **FIFO (First In, First Out)**

The program allows you to select the scheduling algorithm, input process details, and handle resource suspension scenarios.

---

## Features

- **Multiple Algorithms**: Choose from HRRN, SRT, Round Robin, SJF, and FIFO.
- **Dynamic Input**: Specify the number of processes, their arrival times, execution times, and resource usage.
- **Resource Suspension**: Processes that require resources are suspended and executed at the end.
- **User-Friendly Interface**: Simple prompts to guide you through the scheduling process.

---

## Algorithms Overview

### 1. **HRRN (Highest Response Ratio Next)**
   - Prioritizes processes with the highest response ratio.

### 2. **SRT (Shortest Remaining Time)**
   - Preemptive version of SJF; selects the process with the least remaining execution time.

### 3. **Round Robin**
   - Time-sharing algorithm that assigns a fixed time quantum to each process.

### 4. **SJF (Shortest Job First)**
   - Selects the process with the shortest execution time (non-preemptive).

### 5. **FIFO (First In, First Out)**
   - Executes processes in the order they arrive.

---

## Handling Resource Suspension

- If a process requires a resource, it is **suspended** and will be executed after all other processes have finished.

---

## Requirements

- **Python 3.x**
- **Pandas** (for handling process data)

Install dependencies with:
```bash
pip install pandas
```

---
