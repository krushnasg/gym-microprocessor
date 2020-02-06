import numpy as np 
import pandas as pd
import csv
import math
import gym
from gym import error, spaces, utils
from .processor import Processor, Core, Task, ROOM_TEMPERATURE
from .reader import Reader
import matplotlib.pyplot as plt  


class ProcessorEnv():
    metadata = {'render.modes': ['human']}

    def __init__(self, taskFile=None):
        self.processor = Processor(4,80)
        self.time = 0
        #action is a 3-tuple where,
        #   action[0] indicates coreID, 
        #   action[1] indicates the frequency Mode
        #   action[2] indicates the startTime overhead after allocation
        self.action_space = spaces.Box(low=np.array([-1,1]), high=np.array([self.processor.numCores-1, len(self.processor.cores[0].ipc)-1]), dtype=np.int32)
        t=[1,1]
        print(self.processor.numCores-1)
        while(t[0]<2):
            t = self.action_space.sample()
        print("OUT")
        #Observation is a 3xn matrix, where n is the number of cores
        #It represents Temperature, Frequency Mode and the Time unti which a core is occupied.
        # self.observation_space = spaces.Box(low=0, high=80, shape=(3,self.processor.numCores), dtype=np.int32)
        # self.observation_space = spaces.Dict({
        #     "core_prop": spaces.Box(
        #                     low=np.tile(np.array([25,0,0]),(3,1)).transpose(), 
        #                     high=np.tile(np.array([int(self.processor.Tsafe), len(self.processor.cores[0].ipc), np.Infinity]),(3,1)).transpose(), 
        #                     dtype=np.int32),
        #     "upcomingTask": spaces.Box(low=np.array([0,self.time]), high=np.array([100,np.Infinity]), dtype=np.int32)
        #     "time" : spaces.Box(low=0, high= np.Infinity, shape=(1,), dtype=np.int32),
        # })

        self.observation_space = spaces.Box(
            low= np.tile(np.array([25,0,0,0,self.time,0]),(self.processor.numCores,1)).transpose(),
            high= np.tile(np.array([int(self.processor.Tsafe), len(self.processor.cores[0].ipc), np.Infinity, 100, np.Infinity, np.Infinity]),(self.processor.numCores,1)).transpose(),
            dtype= np.int32  
            )
        #for self specified task list
        if not taskFile:
            self.taskList = None
            self.manualInput = False
        else:
            self.taskList = Reader.getTaskList(taskFile=taskFile)
            self.manualInput = True
            print(len(self.taskList))

        self.tempRec = [[] for i in range(self.processor.numCores)]
        self.avgTemp = []

    
    def _getNextTask(self,newSet=False):
        if self.taskList is None:
            if (np.random.random() < 0.2):
                return self._generateRandomTask(newSet=newSet)
            else:
                return None
        else:
            # if not self.taskList:
            #     return None
            if(self.taskList and self.time >= self.taskList[-1].arrivalTime):
                return self.taskList.pop()
            else:
                return None

    #A function to generate a random task
    def _generateRandomTask(self, newSet=False):
        instructionCount = np.random.randint(100) # Instruction count of each task is a random integer between 0 and 100
        arrivalTime = self.time # Arrival Time is a current time
        # Deadline time is minimum execution time + arrival time + a random integer between 0 and 10
        deadlineTime = arrivalTime + (instructionCount//self.processor.cores[0].ipc[-1]) + np.random.randint(10)
        return Task(instructionCount,arrivalTime,deadlineTime,newSet)


    def _temperatureConstraintsSatisfied(self):
        # returns true if temperature at the end of the allocated task execution stays below the critical temperature(Tsafe)
        # return (max((core.temperature 
        #                     + core.temperatureIncrememtRate[0]*max(0,startTime - core.occupiedTill)),
        #                 ROOM_TEMPERATURE) 
        #         + executionTime*core.temperatureIncrememtRate[freqMode] 
        #         < self.processor.Tsafe
        #         )

        flag =True
        for c in self.processor.cores:
            if c.temperature > self.processor.Tsafe:
                flag = False
                return flag
        return flag
        #returns False if avg chip temperature exceeds above tsafe
        avgTemp = sum(c.temperature for c in self.processor.cores)/self.processor.numCores
        return avgTemp < self.processor.Tsafe

    def _timeConstraintsSatisfied(self):
        #returns true if the task execution is feasible at the current freqMode within its deadline time
        # return ((startTime > self.upcomingTask.arrivalTime) and 
        #         (startTime + executionTime <= self.upcomingTask.deadlineTime)
        #         )
        if self.upcomingTask:
            return (self.upcomingTask.deadlineTime > self.time)
        
        return True

    def _roundoff(self, num):
        return num
        if (num%5 > 2):
            return num - (num%5) + 5
        else:
            return num - (num%5)

    def _getObservation(self):        
        '''
        #Define observation as a 2D array consisting of 
        #       temperature, 
        #       freqMode, and 
        #       occupiedTill for all numCores, 
        #       instruction count of upcoming task,
        #       deadline time of upcoming task,
        #       time
        '''
        observation = np.zeros(shape=(6,self.processor.numCores), dtype=np.int32)
        observation[0,:] = np.array([self._roundoff(c.temperature) for c in self.processor.cores])
        observation[1,:] = np.array([c.freqMode for c in self.processor.cores])
        observation[2,:] = np.array([c.occupiedTill for c in self.processor.cores])
        #Define upcomig task as an array consisting of instruction count and the deadline time
       
        if(self.upcomingTask == None):
            observation[3,:] = np.full((1,self.processor.numCores), 0)
            observation[4,:] = np.full((1,self.processor.numCores), 0)
            # task = np.array([0,0])
        else:
            observation[3,:] = np.full((1,self.processor.numCores), self._roundoff(self.upcomingTask.instructionCount))
            observation[4,:] = np.full((1,self.processor.numCores), self.upcomingTask.deadlineTime)
            # task = np.array(self.upcomingTask.instructionCount, self.upcomingTask.deadlineTime)
        
        #Define time in the form of numpy array
        observation[5,:] = np.full((1,self.processor.numCores), self.time)
        # t = np.array([self.time])
        #Define observation as a dictionary composed of above 3 items
        # observation = {"time": t, "core_prop": core_prop, "upcomingTask": task}
        return observation


    def _allocateTask(self, allocatedCoreID, freqMode):
        
        #No action if allocated Core == -1 or if upcoming task is None
        if self.upcomingTask == None:
            return True
                
        #If an action is taken
        if not (allocatedCoreID==-1 or self.upcomingTask == None):
            core = self.processor.cores[allocatedCoreID]
        #if selected core is available
            if core.occupiedTill <= self.time:
                core.switchMode(freqMode)
                executionTime = math.ceil(self.upcomingTask.instructionCount/core.ipc[freqMode])
                startTime = max(core.occupiedTill, self.time) 
                endTime = startTime + executionTime
                core.occupiedTill = endTime
                self.allocation_output.append(
                    {'Task_ID':self.upcomingTask.taskID, 
                    'Core': allocatedCoreID, 
                    'Frequency': freqMode,
                    'Arrival_time': self.upcomingTask.arrivalTime,
                    'Start_time': startTime,
                    'End_time': endTime ,
                    'Deadline_time': self.upcomingTask.deadlineTime,
                    'Execution_time': executionTime
                    }
                    )
                return True
        
                
        return False
    
    

    def _getReward(self, action):
        # 0 reward for no action
        if (action[0] == -1 or self.upcomingTask == None):
            return 0

        core = self.processor.cores[int(action[0])]
        # positive reward for valid allocation
        if core.occupiedTill <= self.time:
            executionTime = math.ceil(self.upcomingTask.instructionCount/core.ipc[int(action[1])])
            estimated_temp = core.temperature + core.temperatureVariation*executionTime
            #Heavy negativ reward for crossing deadline
            deadline_penalty = min(0,10*(self.upcomingTask.deadlineTime - (self.time + executionTime)))
            return self.processor.Tsafe - estimated_temp + deadline_penalty
        #Negative reward for choosing an occupied processor
        else:
            return 0
    
    def _recordTemperature(self):
        avg=0
        for i,c in enumerate(self.processor.cores):
            self.tempRec[i].append(c.temperature)
            avg+=c.temperature
        avg=avg/self.processor.numCores
        self.avgTemp.append(avg)
        return

    def _timelyUpdates(self):
        #Default updates at each time interval
        # print('#################update call' + str(self.time))
        for c in self.processor.cores:
            c.temperature += c.temperatureVariation
            c.temperature = max(ROOM_TEMPERATURE, c.temperature)
            if c.occupiedTill == self.time:
                c.switchMode(0)

        #Record Temperature
        self._recordTemperature()
        return
    
    def graphShow(self):
        x = [i for i,t in enumerate(self.avgTemp)]
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        l1 = ax.plot(x, self.avgTemp, 'y--')
        l2 = ax.plot(x, self.tempRec[0],'g.-')
        l2 = ax.plot(x, self.tempRec[1],'r.-')
        l2 = ax.plot(x, self.tempRec[2],'b.-')
        l2 = ax.plot(x, self.tempRec[3],'c.-')
        ax.legend(labels = ('Avg_temperature','core0','core1','core2','core3'))
        ax.set_title("Chip Temperature Variation")
        ax.set_xlabel('time')
        ax.set_ylabel('Temp (C)')
        # for c in range(self.processor.numCores):
        #     l2 = ax.plot(self.tempRec[c],x,)
        plt.show()

    def reset(self):
        #reset the environment
        self.time=0
        #reset each core
        for c in self.processor.cores:
            c.reset() 
        self.upcomingTask = self._getNextTask(newSet=True)
        self.allocation_output =  []
        observation = self._getObservation()
        # self.tempRec = [[] for i in range(self.processor.numCores)]
        # self.avgTemp = []
        return observation
    
    def step(self, action):
        #action is a 3-tuple where,
        #   action[0] indicates coreID, 
        #   action[1] indicates the frequency Mode
        #   action[2] indicates the startTime overhead after allocation
        allocatedCoreID = int(action[0])
        freqMode = int(action[1])
        
        #allocate the task to proper core based on action
        allocated = self._allocateTask(allocatedCoreID, freqMode)
        #Check if the allocation is feasible in terms of temperature constraints
        if not (self._temperatureConstraintsSatisfied() and self._timeConstraintsSatisfied()):
            done = True
        else:
            done = False
        reward = self._getReward(action)
        #get observation(next state)
        observation = self._getObservation()

        #get next task
        if allocated:        
            self.upcomingTask = self._getNextTask()

        if self.upcomingTask is None or not allocated:
            #update the current time
            self._timelyUpdates() 
            self.time = self.time + 1
            
            if self.manualInput and not self.taskList:
                done = True
        return observation, reward, done, {'info':self.allocation_output}
        
    def render(self, mode='human'):
        observation = self._getObservation()
        rows = ['Temperature', 'Freq Mode', 'Occupied Till', 'Instruction Count', 'Deadline time', 'Current time']
        columns = [f'Core-{i}' for i in range(self.processor.numCores)]
        df = pd.DataFrame(observation, index=rows, columns=columns)
        if (self.time == 0):
          print('\n*****************EPISODE BEGIN********************')
        if (self.upcomingTask):
            print(f'\nUpcoming Task ID: {self.upcomingTask.taskID}, instructionCount:{self.upcomingTask.instructionCount} Atime: {self.upcomingTask.arrivalTime}, Dtime: {self.upcomingTask.deadlineTime}')
            print(f'Current Time: {self.time}')
            print(df)
            print("\n")
        