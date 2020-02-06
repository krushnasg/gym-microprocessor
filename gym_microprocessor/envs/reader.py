import pandas as pd
import numpy as np
from .processor import Task
import matplotlib.pyplot as plt  

class Reader:
    def __init__(self):
        pass

    def getTaskList(taskFile):
        df = pd.read_excel(taskFile,header=0,index_col=None,dtype=float)
        taskList = []
        for index,row in df.iterrows():
            taskList.append(Task(row[0],row[1],row[2]))
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