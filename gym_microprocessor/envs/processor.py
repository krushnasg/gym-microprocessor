import numpy as np
import itertools


ROOM_TEMPERATURE = 25
SAFE_TEMPERATURE = 80
TEMPERATURE_INCREMENT_RATE = [-2, 3, 4]
IPC = [0, 5, 8]

class Processor():
    def __init__(self, numCores, Tsafe):
        self.numCores = numCores
        self.Tsafe = Tsafe
        self.cores = [Core(i) for i in range(numCores)] #defines core position

class Core():
    def __init__(self, coreID, ipc=IPC, switchingOverhead=None, tempIncrRate=TEMPERATURE_INCREMENT_RATE):
        self.coreID = coreID
        self.ipc = ipc
        self.switchingOverhead = switchingOverhead
        self.temperatureIncrememtRate = tempIncrRate
        self.temperature = ROOM_TEMPERATURE
        self.freqMode = 0
        self.occupiedTill = 0
    
    def reset(self):
        self.temperature = ROOM_TEMPERATURE
        self.freqMode = 0
        self.occupiedTill = 0

    def allocateTask(self, task, freqMode, startTime=None):
        if startTime is None:
            startTime = task.arrivalTime
        pass
    

class Task():
    newid = next(itertools.count())
    def __init__(self, instructionCount,arrivalTime, deadlineTime):
        self.instructionCount = instructionCount
        self.arrivalTime = arrivalTime
        self.deadlineTime = deadlineTime
        self.taskID = self.newid