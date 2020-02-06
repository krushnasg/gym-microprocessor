import gym

import csv
import matplotlib.pyplot as plt  

from gym_microprocessor.envs.microprocessor_env import ProcessorEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


def writeOutputFile(filename, data):
        csv_columns = ['Task_ID', 'Core','Frequency','Arrival_time','Start_time','End_time','Deadline_time','Execution_time']
        try:
            with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")

def graph(formula, x_range):  
    x = np.array(x_range)  
    y = eval(formula)
    plt.plot(x, y)  
    plt.show()

# The algorithms require a vectorized environment to run
env =  DummyVecEnv([lambda: ProcessorEnv()])

model = PPO2(MlpPolicy, env, verbose=1,learning_rate=0.00025)
model.learn(total_timesteps=5000)
model.save("ppo2_microprocessor_4")
model = PPO2.load("ppo2_microprocessor_4")

# env = DummyVecEnv([lambda: ProcessorEnv(taskFile='data/example.xlsx')])
env = DummyVecEnv([lambda: ProcessorEnv()])

obs = env.reset()
print ("^^^^^^^^^^^^^^^^^^^RESET")
# env.env_method('graphShow')
# for i in range(200):
done = False
while not done:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print(info[0])
        writeOutputFile('out.csv', info[0]['info'])
        env.env_method('graphShow')
