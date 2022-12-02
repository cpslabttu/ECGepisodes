# Importing the library
import time
import multiprocessing as mp
import psutil
import numpy as np
from requests.packages import target
import matplotlib.pyplot as plt
import ANN_onestep



def monitor(target):

    worker_process = mp.Process(target=target)
    worker_process.start()
    p = psutil.Process(worker_process.pid)
    # log cpu usage of `worker_process` every 10 ms
    cpu_percents = []
    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        time.sleep(0.01)
    worker_process.join()
    return cpu_percents

if __name__ ==  '__main__':
    cpu_percents = monitor(target)
    plt.plot(cpu_percents)
    plt.show()