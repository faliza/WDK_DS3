import multiprocessing
import time
import os
import signal

def cpu_stress():
    while True:
        pass  # max CPU usage

if __name__ == "__main__":
    print("Starting workload...")
    
    # Start multiple processes (one per core)
    processes = []
    for _ in range(multiprocessing.cpu_count()):
        p = multiprocessing.Process(target=cpu_stress)
        p.start()
        processes.append(p)

    time.sleep(10)  # run workload for 10 seconds

    print("Stopping workload...")
    
    for p in processes:
        p.terminate()
        p.join()

    print("System is now idle. Measuring idle power...")
    
    time.sleep(30)  # keep system idle for measurement
