import numpy as np
import itertools


ROOM_TEMPERATURE = 25
SAFE_TEMPERATURE = 80
TEMPERATURE_INCREMENT_RATE = [-2, 3, 6]
POWER_INCREMENT_RATE = [2, 6, 12]#for power modes OFF, LOW and HIGH respectively
SWITCHING_OVERHEAD =[[0, 12, 20],
                     [12, 0, 12],
                     [20, 12, 0]] #row represents current frequency mode and column represents the new freqency mode
IPC = [0, 5, 8]

class Processor():
    def __init__(self, numCores, Tsafe):
        self.numCores = numCores #number of cores in a processor
        self.Tsafe = Tsafe #critical temperature upto which safe operation of the chip can be ensured
        self.cores = [Core(i) for i in range(numCores)] #defines core position

class Core():
    def __init__(self, coreID, ipc=IPC, switchingOverhead=SWITCHING_OVERHEAD, tempIncrRate=TEMPERATURE_INCREMENT_RATE, powerIncrRate=POWER_INCREMENT_RATE):
        self.coreID = coreID
        self.ipc = ipc #an array representing instruction per cycle for each mode
        self.switchingOverhead = switchingOverhead #inter-mode switching overhead in terms of power
        self.temperatureIncrementRate = tempIncrRate #an array determining temperature increment rate according to the processing mode of the core.
        #NOTE: temperature increment is considered linear for now.
        self.powerIncrementRate = powerIncrRate
        self.temperature = ROOM_TEMPERATURE #core temperature, initialised to room temperature
        self.freqMode = 0
        self.temperatureVariation = self.temperatureIncrementRate[self.freqMode]
        self.powerVariation = self.powerIncrementRate[self.freqMode]
        self.occupiedTill = 0   #after a task allocation of a task to a particular core, 
                                #the core will stay occupied until the task finishes execution, and 
                                #no new task can be allocated to this core in the meantime.
    
    def reset(self): #function to reset the core 
        self.temperature = ROOM_TEMPERATURE
        self.switchMode(0)
        self.occupiedTill = 0
        return 
    
    def switchMode(self, mode):#Returns sitching overhead
        overhead = 0
        if (self.switchingOverhead):
            overhead = self.switchingOverhead[self.freqMode][mode]
        self.freqMode = mode
        self.temperatureVariation = self.temperatureIncrementRate[mode]
        self.powerVariation = self.powerIncrementRate[mode]
        
        #NOTE: switching overhead to be added
        return overhead


class Task():
    newid = itertools.count(0) #for auto incrementing the task ids
    def __init__(self, instructionCount,arrivalTime, deadlineTime, newSet=False, taskID=None):
        self.instructionCount = instructionCount
        self.arrivalTime = arrivalTime
        self.deadlineTime = deadlineTime
        if (newSet):
            Task.newid = itertools.count(0) #task id starts from zero for a new set
        if not taskID:
            self.taskID = next(Task.newid)
        else:
            self.taskID = taskID