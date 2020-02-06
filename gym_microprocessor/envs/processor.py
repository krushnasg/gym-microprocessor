import numpy as np
import itertools


ROOM_TEMPERATURE = 25
SAFE_TEMPERATURE = 80
TEMPERATURE_INCREMENT_RATE = [-2, 3, 4]#for power modes OFF, LOW and HIGH respectively
IPC = [0, 5, 8]

class Processor():
    def __init__(self, numCores, Tsafe):
        self.numCores = numCores #number of cores in a processor
        self.Tsafe = Tsafe #critical temperature upto which safe operation of the chip can be ensured
        self.cores = [Core(i) for i in range(numCores)] #defines core position

class Core():
    def __init__(self, coreID, ipc=IPC, switchingOverhead=None, tempIncrRate=TEMPERATURE_INCREMENT_RATE):
        self.coreID = coreID
        self.ipc = ipc #an array representing instruction per cycle for each mode
        self.switchingOverhead = switchingOverhead #inter-mode switching overhead in terms of power
        self.temperatureIncrememtRate = tempIncrRate #an array determining temperature increment rate according to the processing mode of the core.
        #NOTE: temperature increment is considered linear for now.
        self.temperature = ROOM_TEMPERATURE #core temperature, initialised to room temperature
        self.freqMode = 0
        self.temperatureVariation = self.temperatureIncrememtRate[self.freqMode]
        self.occupiedTill = 0   #after a task allocation of a task to a particular core, 
                                #the core will stay occupied until the task finishes execution, and 
                                #no new task can be allocated to this core in the meantime.
    
    def reset(self): #function to reset the core 
        self.temperature = ROOM_TEMPERATURE
        self.switchMode(0)
        self.occupiedTill = 0
        return 
    
    def switchMode(self, mode):
        # print ("call hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" + str(self.coreID) +"\t" + str(mode) + "\t" + str(self.freqMode))
        self.freqMode = mode
        self.temperatureVariation = self.temperatureIncrememtRate[mode]
        #NOTE: power consumption to be added
        return


class Task():
    newid = itertools.count(0) #for auto incrementing the task ids
    def __init__(self, instructionCount,arrivalTime, deadlineTime, newSet=False):
        self.instructionCount = instructionCount
        self.arrivalTime = arrivalTime
        self.deadlineTime = deadlineTime
        if (newSet):
            Task.newid = itertools.count(0) #task id starts from zero for a new set
        self.taskID = next(Task.newid)