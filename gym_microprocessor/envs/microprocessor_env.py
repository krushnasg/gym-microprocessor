import numpy as np 
import pandas as pd
import csv
import math
import gym
from gym import error, spaces, utils
from .processor import Processor, Core, Task, ROOM_TEMPERATURE, SAFE_TEMPERATURE
from .reader import Reader
import matplotlib.pyplot as plt  


class ProcessorEnv():
    metadata = {'render.modes': ['human']}
    
    def __init__(self, taskFile=None):
        self.processor = Processor(4,80)
        self.time = 0
        self.reward_range = (-float('inf'), float('inf'))
        self.power_consumed = 0    
        #action is a 3-tuple where,
        #   action[0] indicates coreID, 
        #   action[1] indicates the frequency Mode
        #   action[2] indicates the startTime overhead after allocation
        self.action_space = spaces.Box(low=np.array([-1,1]), high=np.array([self.processor.numCores-1, len(self.processor.cores[0].ipc)-1]), dtype=np.int32)

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
            print("Generating taskList...")
            self.taskList = Reader.getTaskList(taskFile=taskFile, processor=self.processor)
            self.manualInput = True
            print("taskList generated." + str(len(self.taskList)) + " tasks")

        self.tempRec = [[] for i in range(self.processor.numCores + 1)] # last row will record time
        self.avgTemp = []
        self.powerRec = []
        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.num_episodes = 0
        self.mean_episode_reward = []
        self.mean_episode_length = []
        self.ep_rew = []
        self.ep_instr = []
        self.ep_delay_per_task = []
        self.ep_power_saved = []

    
    def _getNextTask(self,newSet=False):
        if self.taskList is None:
            if (np.random.random() < 0.2):
                return self._generateRandomTask(newSet=newSet)
            else:
                return None
        else:
            # if not self.taskList:
            #     return None
            if(self.taskList):
                raw_task = self.taskList.pop(0)
                Temp_min = SAFE_TEMPERATURE+1
                for c in self.processor.cores:
                    if (Temp_min > c.temperature):
                        Temp_min = c.temperature
                        minCore = c
                v1 = (SAFE_TEMPERATURE - Temp_min)/c.temperatureIncrementRate[-1]
                v2 = (raw_task[1] - raw_task[0])*c.ipc[-1]
                print(raw_task[2])
                v = min(int(v1),int(v2))
                instrCnt = np.random.randint(max(5,v))
                return Task(instrCnt,raw_task[0],raw_task[1],taskID=raw_task[2])
            else:
                return None

    #A function to generate a random task
    def _generateRandomTask(self, newSet=False):
        Temp_min = SAFE_TEMPERATURE+1
        for c in self.processor.cores:
            if (Temp_min > c.temperature):
                Temp_min = c.temperature
                minCore = c
        v = c.ipc[-1]*(SAFE_TEMPERATURE - Temp_min)/c.temperatureIncrementRate[-1]
        instructionCount = np.random.randint(max(5,v)) + 1 # Instruction count of each task is a random integer between 0 and 100
        arrivalTime = self.time # Arrival Time is a current time
        # Deadline time is minimum execution time + arrival time + a random integer between 0 and 10
        low_bound = math.ceil(instructionCount/self.processor.cores[0].ipc[-1])
        high_bound = low_bound + math.ceil(instructionCount/self.processor.cores[0].ipc[1])
        deadlineTime = arrivalTime + np.random.randint(low=low_bound, high=high_bound+5) 
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
        # core = self.processor.cores[allocatedCoreID]
        # executionTime = math.ceil(self.upcomingTask.instructionCount/core.ipc[freqMode])
        # startTime = max(core.occupiedTill, self.time) 

        # return ((startTime > self.upcomingTask.arrivalTime) and 
        #         (startTime + executionTime <= self.upcomingTask.deadlineTime)
        #         )
        if self.upcomingTask:
            return (self.upcomingTask.deadlineTime > self.time)
        
        return True

    def _roundoff(self, num):
        return int(num)
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
                overhead = core.switchMode(freqMode)
                self.power_consumed += overhead

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
                    'Execution_time': executionTime,
                    'Instruction_count': self.upcomingTask.instructionCount
                    }
                    )
                return True
        
                
        return False
    
    

    def _getReward(self, action):
        #EXPT6 

        # 0 reward for no action
        if (action[0] == -1 or self.upcomingTask == None):
            if self.upcomingTask is not None:
                return (self.upcomingTask.arrivalTime - self.time)*10/(self.upcomingTask.deadlineTime - self.upcomingTask.arrivalTime)
            else:
                return 0
        # print('nonero')
        core = self.processor.cores[int(action[0])]
        # reward for valid allocation
        if core.occupiedTill <= self.time:
            executionTime = math.ceil(self.upcomingTask.instructionCount/core.ipc[int(action[1])])
            estimated_temp = core.temperature + core.temperatureVariation*executionTime
            deadline_penalty = min(0,1*(self.upcomingTask.deadlineTime - (self.time + executionTime)))
            overhead = core.switchingOverhead[core.freqMode][int(action[1])]
            power_penalty = ((self.time - self.upcomingTask.arrivalTime)*core.powerVariation #Power consumed before starting of execution 
                            + executionTime * core.powerIncrementRate[int(action[1])] # power consumed during execution
                            +(max(0,self.upcomingTask.deadlineTime - self.time - executionTime) * core.powerIncrementRate[0]) #power consumed after execution
                            + overhead) # switching overhead
            power_saved = (self.upcomingTask.deadlineTime - self.upcomingTask.arrivalTime) * core.powerIncrementRate[-1] - power_penalty
            # reward = 2 * self.upcomingTask.instructionCount + deadline_penalty - power_penalty
            #Temperature component on a scale of 0-10
            temp_component = 10 * (self.processor.Tsafe - estimated_temp)/(self.processor.Tsafe - ROOM_TEMPERATURE)
            #Power component on a scale of 0-10
            power_component = 0
            if power_saved != 0:
                max_power_save = (power_saved 
                                  + power_penalty 
                                  - (self.time - self.upcomingTask.arrivalTime)*core.powerVariation
                                  - (math.ceil(self.upcomingTask.instructionCount/core.ipc[1])) * core.powerIncrementRate[1]
                                  - (max(0,self.upcomingTask.deadlineTime - self.time - math.ceil(self.upcomingTask.instructionCount/core.ipc[1]))) * core.powerIncrementRate[0]
                                  - core.switchingOverhead[core.freqMode][1]
                                  )
                                  
                power_component = 10 * power_saved/max(max_power_save,10)
            reward  = temp_component + power_component + deadline_penalty * 10/(self.upcomingTask.deadlineTime - self.upcomingTask.arrivalTime)

            return reward
        else:
            return (self.upcomingTask.arrivalTime - self.time)*10/(self.upcomingTask.deadlineTime - self.upcomingTask.arrivalTime)


        #EXPT3
        # # 0 reward for no action
        # if (action[0] == -1 or self.upcomingTask == None):
        #     return 0
        # # print('nonero')
        # core = self.processor.cores[int(action[0])]
        # # reward for valid allocation
        # if core.occupiedTill <= self.time:
        #     executionTime = math.ceil(self.upcomingTask.instructionCount/core.ipc[int(action[1])])
        #     deadline_penalty = min(0,10*(self.upcomingTask.deadlineTime - (self.time + executionTime)))
        #     overhead = core.switchingOverhead[core.freqMode][int(action[1])]
        #     power_penalty = overhead + executionTime * core.powerIncrementRate[int(action[1])]
        #     reward = 2 * self.upcomingTask.instructionCount + deadline_penalty - power_penalty

        #     return reward
        # else:
        #     return -1

        #EXPT1
        # if not (self._temperatureConstraintsSatisfied() and self._timeConstraintsSatisfied()):
        #     reward = 0
        # else:
        #     reward = 1
        # return reward

        #EXPT2
        # 0 reward for no action
        if (action[0] == -1 or self.upcomingTask == None):
            return 0
        # print('nonero')
        core = self.processor.cores[int(action[0])]
        # positive reward for valid allocation
        if core.occupiedTill <= self.time:
            # print('here')
            executionTime = math.ceil(self.upcomingTask.instructionCount/core.ipc[int(action[1])])
            estimated_temp = core.temperature + core.temperatureVariation*executionTime
            #Heavy negative reward for crossing deadline
            deadline_penalty = min(0,1*(self.upcomingTask.deadlineTime - (self.time + executionTime)))
            return self.processor.Tsafe - estimated_temp + deadline_penalty
        #Negative reward for choosing an occupied processor
        else:
            return -1
    
    def _recordTemperature(self):
        avg=0
        for i,c in enumerate(self.processor.cores):
            self.tempRec[i].append(c.temperature)
            avg+=c.temperature
        avg=avg/self.processor.numCores
        self.avgTemp.append(avg)
        self.powerRec.append(self.power_consumed)
        self.tempRec[-1].append(self.time)
        return

    def _timelyUpdates(self,t=1):
        #Default updates at each time interval
        # print('#################update call' + str(self.time))
        for c in self.processor.cores:
            c.temperature += t*c.temperatureVariation
            c.temperature = max(ROOM_TEMPERATURE, c.temperature)
            if c.occupiedTill == self.time:
                c.switchMode(0)
            
            self.power_consumed += t*c.powerVariation

        #Record Temperature
        # self._recordTemperature()
        return
    
    def graphShow(self, mode):
        if mode == 'temp':
            x = self.tempRec[-1]
            # fig = plt.figure()
            # ax = fig.add_axes([0,0,1,1])
            # l0 = ax.plot(x, self.powerRec, 'h--')
            plt.plot(x, self.avgTemp, 'y--', label='Avg Temperature')
            plt.plot(x, self.tempRec[0],'g.-', label='core0')
            plt.plot(x, self.tempRec[1],'r.-', label='core1')
            plt.plot(x, self.tempRec[2],'b.-', label='core2')
            plt.plot(x, self.tempRec[3],'c.-', label='core3')
            # ax.legend(labels = ('Avg_temperature','core0','core1','core2','core3'))
            plt.title("Chip Temperature Variation")
            plt.xlabel('time')
            plt.ylabel('Temp (C)')
            plt.legend()
            # for c in range(self.processor.numCores):
            #     l2 = ax.plot(self.tempRec[c],x,)
            plt.show()
        
        elif mode == 'power':
            x = self.tempRec[-1]
            # fig, ax = plt.subplots()
            # ax = fig.add_axes([0,0,1,1])
            plt.plot(x, self.powerRec, 'b.-', label='Power')
            
            
            plt.title("Chip Power Variation")
            plt.xlabel('time')
            plt.ylabel('Power')
            plt.legend()
            # for c in range(self.processor.numCores):
            #     l2 = ax.plot(self.tempRec[c],x,)
            plt.show()
        

    def _monitor(self, observation, reward, done, info):
        # print (reward)
        self.rewards.append(reward)
        if done:
            self.num_episodes += 1
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            # print(self.rewards)
            # print ("Episode Reward {:d};  Episode length {:d}".format(ep_rew,ep_len))
            self.episode_rewards.append(ep_rew)
            self.ep_rew.append(ep_rew)
            self.episode_lengths.append(ep_len)
            if len(self.episode_rewards) == 100:
                self.mean_episode_reward.append(sum(self.episode_rewards)/100)
                self.mean_episode_length.append(sum(self.episode_lengths)/100)
                self.episode_lengths = []
                self.episode_rewards = []
            totalInstr = 0
            delayPerTask = 0
            for t in info[0]['info']:
                totalInstr += t['Instruction_count']
                delayPerTask += max(0,t['End_time'] - t['Deadline_time'])
            if not len(info[0]['info']) == 0:
                delayPerTask = delayPerTask/len(info[0]['info']) == 0 
            runtime = observation[5][0]
            powerSaved = 0
            for c in self.processor.cores:
                powerSaved += runtime * c.powerIncrementRate[-1]
            powerSaved = powerSaved - self.power_consumed
            powerSaved = powerSaved/runtime
            self.ep_instr.append(totalInstr)
            self.ep_power_saved.append(powerSaved)
            self.ep_delay_per_task.append(delayPerTask)

    def seed(self, seed=None):
        seed = [1]
        return seed

        
    def reset(self):
        #Setting a random state
        # self.upcomingTask = self._getNextTask(newSet=True)
        self.upcomingTask = self._getNextTask()
        if self.taskList is None:
            self.time=0
        else:
            self.time = self.upcomingTask.arrivalTime
        for c in self.processor.cores:
            c.temperature = np.random.randint(ROOM_TEMPERATURE,SAFE_TEMPERATURE)
            c.switchMode(np.random.randint(3))
            c.occupiedTill = 0
        self.power_consumed = 0
        # self.upcomingTask = self._getNextTask(newSet=True)
        self.allocation_output =  []
        observation = self._getObservation()
        self.rewards = []
        return observation


        #reset the environment...For testing
        self.time=0
        #reset each core
        for c in self.processor.cores:
            c.reset() 
        self.power_consumed = 0
        self.upcomingTask = self._getNextTask(newSet=True)
        self.allocation_output =  []
        observation = self._getObservation()
        # self.tempRec = [[] for i in range(self.processor.numCores)]
        # self.avgTemp = []
        # self._recordTemperature()
        self.rewards = []
        # self.episode_rewards = []
        # self.episode_lengths = []
        return observation
    
    def step(self, action):
        #action is a 3-tuple where,
        #   action[0] indicates coreID, 
        #   action[1] indicates the frequency Mode
        #   action[2] indicates the startTime overhead after allocation
        allocatedCoreID = int(action[0])
        freqMode = int(action[1])
        reward = self._getReward(action)
        #allocate the task to proper core based on action
        allocated = self._allocateTask(allocatedCoreID, freqMode)
        #Check if the allocation is feasible in terms of temperature constraints
        if not (self._temperatureConstraintsSatisfied() and self._timeConstraintsSatisfied()):
            done = True
        else:
            done = False
        
        #get observation(next state)
        observation = self._getObservation()

        #get next task
        if allocated:        
            self.upcomingTask = self._getNextTask()

        if (self.upcomingTask is not None) and self.time < self.upcomingTask.arrivalTime:
            while(self.time < self.upcomingTask.arrivalTime):
                dt = self.upcomingTask.arrivalTime
                for c in self.processor.cores:
                    if c.occupiedTill > self.time:
                        dt = min(dt,c.occupiedTill)
                dt = dt - self.time
                self.time += dt
                self._timelyUpdates(t=dt)

        elif self.upcomingTask is None or not allocated:
            #update the current time
            self._timelyUpdates() 
            self.time = self.time + 1
            
            if self.manualInput and not self.taskList:
                done = True
        
        self._monitor(observation, reward, done,[{'info':self.allocation_output}])
        return observation, reward, done, {'info':self.allocation_output}
        
    def render(self, mode='human'):
        # if (self.time == len(self.tempRec)):
        # print (self.time, len(self.tempRec[0]))
        if (self.time == len(self.tempRec[0])):
            self._recordTemperature()
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
        