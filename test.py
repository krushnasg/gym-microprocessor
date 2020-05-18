import os
import argparse
import csv
import gym

from gym_microprocessor.envs.microprocessor_env import ProcessorEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

def writeOutputFile(filename, data):
        csv_columns = ['Task_ID', 'Core','Frequency','Arrival_time','Start_time','End_time','Deadline_time','Execution_time','Instruction_count']
        try:
            with open(filename, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
                for data in data:
                    writer.writerow(data)
        except IOError:
            print("I/O error")


parser = argparse.ArgumentParser()
parser.add_argument("model_file",help="model file")
args = parser.parse_args()
model_file = args.model_file

num_test = 1000

env = DummyVecEnv([lambda: ProcessorEnv()])
model = PPO2.load(model_file)
model.set_env(env)

obs = env.reset()
done = False
episode_count = 0
taskID = -1
cumInstructions = 0
avgPowerPerInstr = 0
avgDelayPerTask = 0
skip = 0
while not episode_count == num_test:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        print(info[0])
        episode_count += 1
        totalInstr = 0
        delayPerTask = 0
        for t in info[0]['info']:
            print(t)
            totalInstr += t['Instruction_count']
            delayPerTask += max(0,t['End_time'] - t['Deadline_time'])
        if(totalInstr == 0):
            skip += 1
            # avgPowerPerInstr += env.get_attr('powerRec')[0][-1]    
            # delayPerTask = delayPerTask
        else:
            avgPowerPerInstr += env.get_attr('powerRec')[0][-1]/totalInstr
            delayPerTask = delayPerTask/len(info[0]['info'])

        cumInstructions += totalInstr
        avgDelayPerTask += delayPerTask
        # writeOutputFile('out.csv', info[0]['info'])
        # env.env_method('graphShow', 'power')
        # env.env_method('graphShow', 'temp')
        
avgPowerPerInstr = avgPowerPerInstr/(num_test - skip)
cumInstructions = cumInstructions/(num_test - skip)
avgDelayPerTask = avgDelayPerTask/(num_test - skip)
print("Mean Instruction Count per episode = \t" + str(cumInstructions))
print("avgPowerPerInstr = \t\t" + str(avgPowerPerInstr))
# print("cumInstructions = " + str(cumInstructions))
print("avgDelayPerTask = \t\t" + str(avgDelayPerTask))
print("skips = \t\t" + str(skip))