import pandas as pd
import numpy as np
from .processor import Task,ROOM_TEMPERATURE,SAFE_TEMPERATURE, IPC
import matplotlib.pyplot as plt  

class Reader:
    def __init__(self):
        pass

    def getTaskList(taskFile, processor):
        df = pd.read_csv(taskFile,header=0,index_col=None)
        taskList = []
        for index,row in df.iterrows():
            taskList.append([row[3],row[4],index])
        return taskList

    def graph(formula, low, high): 
        # plt.contour(x, y, x**2+y**2-9)
        # return 
        x = np.arange(low, high)  
        y = eval('x**2')
        plt.plot(x, x*x)  
        plt.show()

    def showGraph():
        plt.show()
        return 