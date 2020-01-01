import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

NUM_CORES = 4
NUM_FREQUENCY_MODES = 3
FREQUENCY_MODES = [0,12,25]
TEMP_ROOM = 25
TEMP_SAFE = 80
TEMP_INCREMENT_RATE = 2

class MicroprocessorEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self,num_cores=NUM_CORES):
    super(MicroprocessorEnv, self).__init__()
    self.num_cores = num_cores
    self.action_space = spaces.Discrete(self.num_cores)
    self.observation_space = spaces.Box(low=TEMP_ROOM, high=TEMP_SAFE, shape=(self.num_cores,),dtype=np.int32)
    self.taskNo = 0
      # 'Task' attributes should be included in state
     
      # 'Frequency': spaces.Box(low=0, high=NUM_FREQUENCY_MODES, shape=(self.num_cores,),dtype=np.int32) , 
      # 'Utilization': spaces.Box(low=0.0, high=100.0, shape(self.num_cores,), dtype=np.float32)

  def checkGameOver(self):
    for i in self.temperature:
      if (i>TEMP_SAFE):
        return True
    return False

  def step(self, action):
    #action changes the Frequency, utilization distribution of cores and 
    reward = self.upcomingTaskTime * TEMP_INCREMENT_RATE
    self.temperature[action] += self.upcomingTaskTime * TEMP_INCREMENT_RATE
    self.upcomingTaskTime = np.random.randint(1,10)
    done = False
    if (self.checkGameOver()):
      done = True
    else:
      self.taskNo += 1

    obs = np.array(self.temperature)
    
    return obs, reward, done, {}
    
 
  def reset(self):
    self.temperature = [TEMP_ROOM for i in range(self.num_cores)]
    self.upcomingTaskTime = np.random.randint(1,10)  
 
    obs = np.array(self.temperature)
    self.taskNo = 0
    return obs
    
  def render(self, mode='human'):
    if (self.checkGameOver()):
      print ("GAME OVER")
    else:
      print (self.upcomingTaskTime,self.taskNo,self.temperature)
      print ("..................")
    
  def close(self):
    pass
