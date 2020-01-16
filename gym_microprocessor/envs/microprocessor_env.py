import numpy as np 
import pandas as pd
import math
import gym
from gym import error, spaces, utils
from .processor import Processor, Core, Task, ROOM_TEMPERATURE

class ProcessorEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.processor = Processor(4,80)
        #action is a 3-tuple where,
        #   action[0] indicates coreID, 
        #   action[1] indicates the frequency Mode
        #   action[2] indicates the startTime overhead after allocation
        self.action_space = spaces.Box(low=np.array([0,1,0]), high=np.array([self.processor.numCores-1, len(self.processor.cores[0].ipc)-1, 5]), dtype=np.int32)
        #Observation is a 3xn matrix, where n is the number of cores
        #It represents Temperature, Frequency Mode and the Time unti which a core is occupied.
        self.observation_space = spaces.Box(low=0, high=80, shape=(3,self.processor.numCores), dtype=np.int32)

    #A function to generate a random task
    def _generateRandomTask(self, newSet=False):
        instructionCount = np.random.randint(100) # Instruction count of each task is a random integer between 0 and 100
        arrivalTime = self.time + np.random.randint(5) # Arrival Time is a random time between current time and current time + 5 units
        # Deadline time is minimum execution time + arrival time + a random integer between 0 and 10
        deadlineTime = arrivalTime + (instructionCount//self.processor.cores[0].ipc[-1]) + np.random.randint(10)
        return Task(instructionCount,arrivalTime,deadlineTime,newSet)

    def _temperatureConstraintsSatisfied(self, core, startTime, executionTime, freqMode):
        #returns true if temperature at the end of the allocated task execution stays below the critical temperature(Tsafe)
        return (max((core.temperature 
                            + core.temperatureIncrememtRate[0]*max(0,startTime - core.occupiedTill)),
                        ROOM_TEMPERATURE) 
                + executionTime*core.temperatureIncrememtRate[freqMode] 
                < self.processor.Tsafe
                )

    def _timeConstraintsSatisfied(self, core, startTime, executionTime, freqMode):
        #returns true if the task execution is feasible at the current freqMode within its deadline time
        return ((startTime > self.upcomingTask.arrivalTime) and 
                (startTime + executionTime <= self.upcomingTask.deadlineTime)
                )

    def _getObservation(self):
        #Define Observation as a 2D array consisting of temperature, freqMode, and occupiedTill for all numCores
        observation = np.zeros(shape=(3,self.processor.numCores), dtype=np.int32)
        observation[0,:] = np.array([c.temperature for c in self.processor.cores])
        observation[1,:] = np.array([c.freqMode for c in self.processor.cores])
        observation[2,:] = np.array([c.occupiedTill for c in self.processor.cores])
        return observation


    def _allocateTask(self, allocatedCoreID, freqMode, startTimeOverhead):
        core = self.processor.cores[allocatedCoreID] 
        executionTime = math.ceil(self.upcomingTask.instructionCount/core.ipc[freqMode])
        startTime = max(core.occupiedTill, self.upcomingTask.arrivalTime) + startTimeOverhead
        endTime = startTime + executionTime

        #Check if the allocation is feasible
        if not (self._temperatureConstraintsSatisfied(core, startTime, executionTime, freqMode) and 
            self._timeConstraintsSatisfied(core, startTime, executionTime, freqMode)):
            done = True
            reward = 0
            return done, reward
        #If feasible, we allocate the task to the core at given frequency mode
        #Core cools down when not in execution
        core.temperature += core.temperatureIncrememtRate[0]*max(0,startTime - core.occupiedTill) 
        core.temperature = max(core.temperature, ROOM_TEMPERATURE)
        #Core changes its frequency mode to execute the newly allocated task
        core.freqMode = freqMode
        #Core heats while a task is being executed
        core.temperature += executionTime*core.temperatureIncrememtRate[freqMode]
        #Core is occupied until the task finishes its execution        
        core.occupiedTill = endTime
        #A reward of farness from critical temperature is given
        reward = self.processor.Tsafe - core.temperature 
        done = False
        return done, reward


    def reset(self):
        #reset the environment
        self.time=0
        #reset each core
        for c in self.processor.cores:
            c.reset() 
        self.upcomingTask = self._generateRandomTask(newSet=True)
        observation = self._getObservation()
        return observation
    
    def step(self, action):
        #action is a 3-tuple where,
        #   action[0] indicates coreID, 
        #   action[1] indicates the frequency Mode
        #   action[2] indicates the startTime overhead after allocation
        allocatedCoreID = int(action[0])
        freqMode = int(action[1])
        startTimeOverhead = int(action[2])

        #allocate the task to proper core based on action
        done, reward = self._allocateTask(allocatedCoreID, freqMode, startTimeOverhead)
        #get observation(next state)
        observation = self._getObservation()

        #update the current time
        self.time = self.upcomingTask.arrivalTime
        #get next task
        self.upcomingTask = self._generateRandomTask()
        return observation, reward, done, {}
        
    def render(self, mode='human'):
        observation = self._getObservation()
        rows = ['Temperature', 'Freq Mode', 'Occupied Till']
        columns = [f'Core-{i}' for i in range(self.processor.numCores)]
        df = pd.DataFrame(observation, index=rows, columns=columns)
        if (self.upcomingTask.taskID == 0):
          print('\n*****************EPISODE BEGIN********************')
        print(f'\nTask ID: {self.upcomingTask.taskID}, instructionCount:{self.upcomingTask.instructionCount} Atime: {self.upcomingTask.arrivalTime}, Dtime: {self.upcomingTask.deadlineTime}')
        print(f'Current Time: {self.time}')
        print(df)
        