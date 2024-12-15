import pandas as pd
import matplotlib.pyplot as plt


# --------------------------- Round Robin ------------------------------- #
def round_robin(processes, psw):
    """
    Round Robin is a pre-emptive scheduling algorithm which gives a fixed time slice to each process
    and moves to next, it keeps a record of how much time is utilized by each process through psw dictionary
    :param processes: List of process
    :param psw: psw dictionary
    :return: Process table and Gantt Chart
    """
    i = 0  # Current process index
    clock = 0  # Total elapsed time
    check = 0
    suspended = []
    suspend = 0

    # initializing dataframe
    process_data = pd.DataFrame(columns=['PID', 'Arrival Time', 'Execution Time', 'PSW'])
    for process in processes:
        process_data = pd.concat([process_data, pd.DataFrame({
            'PID': [process['PID']],
            'Arrival Time': [process['arrival_time']],
            'Execution Time': [process['execution_time']],
            'PSW': [psw[process['PID']]]
        })], ignore_index=True)
    current_process_data = pd.DataFrame(columns=['PID', 'Instruction 1', 'Instruction 2', 'PSW', 'Clock'])

    # Print initial state
    print(f"Initial state at clock {clock}:\n", process_data)

    # process scheduling
    while processes:  # Run until no processes left
        curr_process = processes[i]
        index = psw[curr_process['PID']]  # Get the process's current index from PSW
        check += 1
        if curr_process["arrival_time"] <= clock:  # check if the process has arrived or not
            check = 0
            # If a process utilizes a resource and has not been suspended before
            if curr_process.get("resource") == 1 and suspend != 2:
                suspended.append(curr_process)
                processes.remove(curr_process)
                suspend = 1
                i -= 1
            else:
                # If more than one instruction remains
                if index + 1 < curr_process['execution_time']:
                    inc1 = curr_process['process'][index]
                    inc2 = curr_process['process'][index + 1]
                    psw[curr_process['PID']] += 2
                    # Print updated state at each tick
                    print(f"\nUpdated state at clock {clock}:\n", process_data)
                    detail1(curr_process, inc1, inc2, psw)
                    clock += 2

                    # Update the current process data with running instructions
                    current_process_data = pd.DataFrame({
                        'PID': [curr_process['PID']],
                        'Instruction 1': [inc1],
                        'Instruction 2': [inc2],
                        'PSW': [psw[curr_process['PID']]],
                        'Clock': [clock]
                    })
                    print(current_process_data)

                else:  # If only one instruction remains
                    inc1 = curr_process['process'][index]
                    psw[curr_process['PID']] += 1
                    inc2 = None
                    # Print updated state at each tick
                    print(f"\nUpdated state at clock {clock}:\n", process_data)
                    detail1(curr_process, inc1, inc2, psw)
                    clock += 1

                    # Update the current process data with running instructions
                    current_process_data = pd.DataFrame({
                        'PID': [curr_process['PID']],
                        'Instruction 1': [inc1],
                        'Instruction 2': [inc2],
                        'PSW': [psw[curr_process['PID']]],
                        'Clock': [clock]
                    })
                    print(current_process_data)

                # If the process is finished (PSW matches its execution time)
                if psw[curr_process['PID']] == curr_process['execution_time']:
                    print(f"Process {curr_process['PID']} Completed!")
                    processes.pop(i)  # Remove the process from the queue
                    i -= 1

        # Update the DataFrame row for the current process
        process_data.loc[process_data['PID'] == curr_process['PID'], 'PSW'] = psw[curr_process['PID']]

        # Move to the next process or loop back to the beginning
        if i < len(processes) - 1:
            i += 1
        else:
            i = 0

        if check == len(processes):
            check = 0
            clock += 1

        # After all processes have finished, handle suspended processes
        if not processes and suspend == 1:
            suspend = 2
            while suspended:
                curr_process = suspended.pop(0)
                processes.append(curr_process)


# ----------------------------- HRRN -------------------------------------#
def HighestResponseRatioNext(processes):
    """
    Highest Response Ratio Next is a Non pre-emptive scheduling algorithm which checks for a process
    with lowest response ratio  which is calculated by a math function and selects the process with the
    highest
    :param processes: List of processes
    :return: Process table and Gantt chart
    """
    tick = 0  # Tracks time
    wait = {}  # Store waiting times for each process
    not_completed_process = list(processes)  # List of unprocessed processes
    suspend = 0
    suspended = []
    ready_queue = []  # Processes ready to execute
    highest_ratio = []

    # Initialize DataFrame to hold process execution information
    columns = ['Process', 'Arrival Time', 'Execution Time', 'Start Time', 'Wait Time', 'Turnaround Time', 'Finish Time',
               'Response Time']
    process_df = pd.DataFrame(columns=columns)

    # Initialize waiting time and other details for each process
    for process in processes:
        wait[process["PID"]] = 0  # Initialize waiting time to 0 for all processes
        process['start_time'] = None
        process['wait_time'] = None
        process['turnaround_time'] = None
        process['finish_time'] = None
        process['response_time'] = None  # Initialize response time to None

    # Main loop: runs until all processes are completed
    while not_completed_process or ready_queue:
        # Add processes to ready queue when their arrival time is equal to the current tick
        for process in not_completed_process[:]:
            if process["arrival_time"] <= tick:  # <= to catch already arrived processes
                ready_queue.append(process)
                not_completed_process.remove(process)
        # If ready queue has processes, calculate HRRN and execute the highest ratio process
        if ready_queue:
            highest_ratio.clear()  # Clear ratios for recalculation

            # Calculate HRRN for each process in the ready queue
            for process in ready_queue:
                waiting_time = tick - process["arrival_time"]
                response_ratio = (waiting_time + process["execution_time"]) / process["execution_time"]
                highest_ratio.append((response_ratio, process))

            # Sort by the highest response ratio (descending order)
            highest_ratio.sort(reverse=True, key=lambda x: x[0])
            selected_process = highest_ratio[0][1]  # Pick the process with the highest HRRN

            print(f"Executing Process PID: {selected_process['PID']} at tick {tick} with HRRN: {highest_ratio[0][0]}")

            # Simulate execution (here, we assume the process runs to completion)
            tick += selected_process["execution_time"]  # Increment time by the process's burst time
            ready_queue.remove(selected_process)  # Remove from ready queue after execution

            # Calculate the times for the selected process
            selected_process['start_time'] = max(tick - selected_process['execution_time'],
                                                 selected_process['arrival_time'])
            selected_process['wait_time'] = selected_process['start_time'] - selected_process['arrival_time']
            selected_process['finish_time'] = tick
            selected_process['turnaround_time'] = selected_process['finish_time'] - selected_process['arrival_time']

            # Response time is the time from the arrival to the first time the process starts
            if selected_process['response_time'] is None:
                selected_process['response_time'] = selected_process['start_time'] - selected_process['arrival_time']

            # Print the results for this process
            print(
                f"Process {selected_process['PID']} finished. Waiting Time: {selected_process['wait_time']},"
                f" Turnaround Time: {selected_process['turnaround_time']}")

            # Create a new DataFrame row for the selected process
            new_row = pd.DataFrame({
                'PID': [selected_process['PID']],
                'Arrival Time': [selected_process['arrival_time']],
                'Execution Time': [selected_process['execution_time']],
                'Start Time': [selected_process['start_time']],
                'Wait Time': [selected_process['wait_time']],
                'Turnaround Time': [selected_process['turnaround_time']],
                'Finish Time': [selected_process['finish_time']],
                'Response Time': [selected_process['response_time']]
            })

            # Append the new row to the DataFrame using pd.concat()
            process_df = pd.concat([process_df, new_row], ignore_index=True)

        # If no process is ready yet, simply increment time
        else:
            tick += 1

    # Print final DataFrame
    print("Final process details:\n", process_df)
    process_df.to_csv("Process_data.csv")
    plot_gantt_chart(process_df)

# ------------------------------------------- SJF --------------------------------------------------#


def SJF(processes):
    """
    Shortest Job First is a non Pre-emptive Scheduling algorithm which Checks for the process
    with least number of instructions(jobs)
    :param processes: List of Processes
    :return: Process table and a Gantt chart
    """
    print(processes)
    suspend = 0
    suspended = []
    arrived = []
    clock = 0

    # DataFrame to store process details
    process_data = pd.DataFrame(columns=['PID', 'Arrival Time', 'Execution Time', 'Start Time', 'Wait Time',
                                         'Turnaround Time', 'Finish Time', 'Response Time'])

    while processes or arrived:

        i = 0
        while i < len(processes):
            if processes[i]["arrival_time"] <= clock:
                arrived.append(processes[i])
                processes.remove(processes[i])
                continue
            i += 1

        curr_process = None
        if arrived:
            ExeT = float('inf')
            for e in arrived:  # process with least execution time
                if e["execution_time"] < ExeT:
                    curr_process = e
                    ExeT = e["execution_time"]

            if curr_process.get("resource") == 1 and suspend != 2:
                suspended.append(curr_process)
                arrived.remove(curr_process)  # Remove the current process
                suspend = 1
            else:
                print("------------------------------------SJF-----------------------------------")
                print(f"Executing {curr_process['PID']} with instructions {curr_process['process']}")

                # Initialize start time if it's not set
                if 'start_time' not in curr_process or curr_process['start_time'] is None:
                    curr_process['start_time'] = clock  # Set start time for the first execution
                    curr_process['response_time'] = curr_process['start_time'] - curr_process['arrival_time']

                arrived.remove(curr_process)
                clock += curr_process["execution_time"]

                # Calculate times
                curr_process['finish_time'] = clock
                curr_process['turnaround_time'] = curr_process['finish_time'] - curr_process['arrival_time']
                curr_process['wait_time'] = curr_process['turnaround_time'] - curr_process['execution_time']

                # Append the current process details to the DataFrame
                process_data = pd.concat([process_data, pd.DataFrame({
                    'PID': [curr_process['PID']],
                    'Arrival Time': [curr_process['arrival_time']],
                    'Execution Time': [curr_process['execution_time']],
                    'Start Time': [curr_process['start_time']],
                    'Wait Time': [curr_process['wait_time']],
                    'Response Time': [curr_process['response_time']],
                    'Turnaround Time': [curr_process['turnaround_time']],
                    'Finish Time': [curr_process['finish_time']]
                })], ignore_index=True)

            if not processes and not arrived and suspend == 1:
                suspend = 2
                while suspended:
                    curr_process = suspended.pop(0)  # Get the next suspended process
                    arrived.append(curr_process)  # Add it back to processes
        else:
            clock += 1

    # Print final DataFrame
    print("Final process details:\n", process_data)
    process_data.to_csv("Process_data.csv")
    plot_gantt_chart(process_data)


# -------------------------FIFO------------------------#
def FIFO(processes):
    """
    First In First Out is a Non Pre-emptive Scheduling algo which runs whichever process comes first and projects a
    process table and a Gantt Chart
    :param processes: List of Processes
    :return: A process Table and a Gantt Chart
    """
    print(processes)

    suspend = 0
    suspended = []
    clock = 0  # Initialize clock to keep track of time

    # Initialize DataFrame for storing process data
    process_data = pd.DataFrame(
        columns=['PID', 'Arrival Time',  'Execution Time', 'Start Time', 'Finish Time', 'Wait Time', 'Turnaround Time',
                 'Response Time'])

    while processes:
        curr_process = None
        arr = 999999

        # Find the process with the earliest arrival time
        for e in processes:
            if e["arrival_time"] < arr:
                arr = e["arrival_time"]
                curr_process = e

        # Check if the process requires a resource and needs to be suspended
        if curr_process.get("resource") == 1 and suspend != 2:
            suspended.append(curr_process)
            processes.remove(curr_process)  # Remove the current process
            suspend = 1
            continue  # Skip to the next iteration

        # Process is executing
        print("------------------------------------FIFO-----------------------------------")
        print(f"Executing {curr_process['PID']} with instructions {curr_process['process']}")

        # Compute the start time (current clock or the arrival time, whichever is later)
        start_time = max(clock, curr_process['arrival_time'])
        finish_time = start_time + curr_process['execution_time']

        # Calculate other metrics
        wait_time = start_time - curr_process['arrival_time']
        turnaround_time = finish_time - curr_process['arrival_time']
        response_time = wait_time  # In FIFO, response time is the same as wait time

        # Update the DataFrame with process info
        process_data = pd.concat([process_data, pd.DataFrame({
            'PID': [curr_process['PID']],
            'Arrival Time': [curr_process['arrival_time']],
            'Start Time': [start_time],
            'Execution Time': [curr_process['execution_time']],
            'Finish Time': [finish_time],
            'Wait Time': [wait_time],
            'Response Time': [response_time],
            'Turnaround Time': [turnaround_time]
        })], ignore_index=True)

        # Update the clock to reflect the finish time of the current process
        clock = finish_time

        # Remove the executed process from the list
        processes.remove(curr_process)

        # If all processes are done, check if there are suspended processes
        if not processes and suspend == 1:
            suspend = 2
            while suspended:
                curr_process = suspended.pop(0)  # Get the next suspended process
                processes.append(curr_process)  # Add it back to processes

    # Print the final DataFrame
    print("\nFinal Process Data:")
    print(process_data)
    process_data.to_csv("Process_data.csv")
    plot_gantt_chart(process_data)


# --------------------------------- SRT ------------------------------------ #
def SRT(processes, psw):
    """
    Shortest Remaining Time scheduling algorithm is a pre-emptive scheduling algo which checks for process
    with the shortest remaining time each tick
    :param processes: List of Processes
    :param psw: Psw Dictionary
    :return: A Process table and a Gantt chart
    """
    print(processes)

    # DataFrame to store process execution data (including additional fields)
    process_data = pd.DataFrame(columns=[
        'PID', 'Start Time', 'Execution Time', 'Wait Time', 'Arrival Time',
        'Turnaround Time', 'Response Time', 'Finish Time'])

    suspend = 0
    suspended = []
    arrived = []
    clock = 0
    quantum = 0

    # Dictionary to track first start times for response time calculation
    first_start_time = {}

    # Dictionary to store the cumulative execution time for each process
    exec_time_tracker = {}

    while processes or arrived:

        # Move processes from "processes" to "arrived" as they arrive
        i = 0
        while i < len(processes):
            if processes[i]["arrival_time"] <= clock:
                arrived.append(processes[i])
                processes.remove(processes[i])
                continue
            i += 1

        curr_process = None
        if arrived:
            ExeT = 9999999
            # Find the process with the least remaining execution time
            for e in arrived:
                remaining_time = e["execution_time"] - psw[e["PID"]]
                if remaining_time < ExeT:
                    curr_process = e
                    ExeT = remaining_time

            # Check if the process needs to be suspended
            if curr_process.get("resource") == 1 and suspend != 2:
                suspended.append(curr_process)
                arrived.remove(curr_process)  # Remove the current process
                suspend = 1

            else:
                # Process execution: log start time
                start_time = clock
                index = psw[curr_process['PID']]
                print("------------------------------------SRT-----------------------------------")
                print(f"-----------------------------------{quantum}-----------------------------")
                print(f"Executing {curr_process['PID']} with instruction {curr_process['process'][index]}")

                # Execute one instruction (one quantum)
                psw[curr_process['PID']] += 1
                clock += 1
                quantum += 1

                # Track the first start time for response time calculation
                if curr_process['PID'] not in first_start_time:
                    first_start_time[curr_process['PID']] = start_time

                # Increment cumulative execution time for the current process
                if curr_process['PID'] not in exec_time_tracker:
                    exec_time_tracker[curr_process['PID']] = 0
                exec_time_tracker[curr_process['PID']] += 1

                # After one instruction, log finish time in the DataFrame
                finish_time = clock

                # Append or update the DataFrame
                process_data = pd.concat([process_data, pd.DataFrame({
                    'PID': [curr_process['PID']],
                    'Start Time': [start_time],
                    'Finish Time': [finish_time],
                    'Arrival Time': [curr_process['arrival_time']],
                    'Execution Time': [exec_time_tracker[curr_process['PID']]],
                    'Turnaround Time': [finish_time - curr_process['arrival_time']],
                    'Response Time': [first_start_time[curr_process['PID']] - curr_process['arrival_time']],
                    'Wait Time': [(finish_time - curr_process['arrival_time']) - exec_time_tracker[curr_process['PID']]]
                })], ignore_index=True)

                # If the process is completed, remove it from the queue
                if psw[curr_process['PID']] == curr_process['execution_time']:
                    print(f"Process {curr_process['PID']} Completed!")
                    arrived.remove(curr_process)

            # If no other processes are left but there are suspended processes, resume them
            if not processes and not arrived and suspend == 1:
                suspend = 2
                while suspended:
                    curr_process = suspended.pop(0)  # Get the next suspended process
                    arrived.append(curr_process)  # Add it back to processes
        else:
            # If no process is ready, increment the clock (idle time)
            clock += 1

    # Print the final DataFrame
    print("\nFinal Process Execution Data:")
    print(process_data)
    process_data.to_csv("Process_data.csv")
    plot_gantt_chart(process_data)


# ----------------------- printing ------------------------- #
def detail1(process, inc1, inc2, psw):
    print("------------------------------------2 time Quantum----------------------------------")
    print(f"Executing {process['PID']} with instructions {inc1} and {inc2}")
    print(process)
    print(psw)


def detail2(process, inc, psw):
    print("------------------------------------2 time Quantum----------------------------------")
    print(f"Executing {process['PID']} with instructions {inc}")
    print(process)
    print(psw)


def detail3(process):
    print("------------------------------------2 time Quantum----------------------------------")
    print(f"actual {process}")


def plot_gantt_chart(df):
    """
    Projects a Gantt chart of a selected scheduling algorithm
    :param df: Pandas DataFrame
    :return: A projection of Gantt chart
    """
    # Ensure that 'Finish Time' and 'Start Time' are numeric
    df['Start Time'] = pd.to_numeric(df['Start Time'])
    df['Finish Time'] = pd.to_numeric(df['Finish Time'])

    # Calculate the duration for each process
    df['Duration'] = df['Finish Time'] - df['Start Time']

    # Define a color map for distinct processes
    color_map = plt.get_cmap('tab10', len(df))  # Get distinct colors from a colormap

    # Create a new figure for the Gantt chart
    plt.figure(figsize=(10, 6))

    # Plot each process as a horizontal bar
    for i, row in df.iterrows():
        plt.barh(row['PID'], row['Duration'], left=row['Start Time'], color=color_map(i), edgecolor='black',
                 label=row['PID'])

    # Adding details to the chart
    plt.xlabel('Time')
    plt.ylabel('Processes')
    plt.title('Gantt Chart of Processes')
    plt.xticks(range(0, int(df['Finish Time'].max()) + 1))  # Set x-ticks based on finish time
    plt.yticks(df['PID'])  # Set y-ticks to show process IDs
    plt.legend(title='Process ID')
    plt.grid(axis='x')

    # Show the Gantt chart
    plt.tight_layout()
    plt.show()


# --------------------------------------Scheduler ------------------------------- #
def run_scheduler(processes, psw, algo):
    """
    Scheduler takes the process list, psw dictionary and asks for which scheduling algorithm to run
    and accordingly selects the function
    :param processes: takes input a list of processes
    :param psw: takes input a psw dictionary
    :param algo: what scheduling algorithm needed to be implemented
    Available Algorithms :
     - Round Robin
     - Highest Response Ratio Next
     - First In First Out (FIFO)
     - Shortest Job First (SJF)
     - Shortest Remaining Time (SRT)
    :return: A table and a gantt chart
    """
    if algo == "SRT":
        SRT(processes, psw)
    elif algo == "FIFO":
        FIFO(processes)
    elif algo == "HRRN":
        HighestResponseRatioNext(processes)
    elif algo == "SJF":
        SJF(processes)
    elif algo == "roundrobin":
        round_robin(processes, psw)


