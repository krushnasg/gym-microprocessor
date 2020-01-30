import pandas as pd
import numpy as np
from .processor import Task

class Reader:
    def __init__(self):
        pass

    def getTaskList(taskFile):
        df = pd.read_excel(taskFile,header=0,index_col=None,dtype=float)
        taskList = []
        for index,row in df.iterrows():
            taskList.append(Task(row[0],row[1],row[2]))
        return taskList