import os
import argparse
import csv
import gym

from gym_microprocessor.envs.microprocessor_env import ProcessorEnv
from gym_microprocessor.envs.test_env import TestEnv
from stable_baselines.deepq import MlpPolicy

# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2,DQN
import numpy as np
import wandb

wandb.init( project="test-gym-microprocessor")
wandb.config["more"] = "custom"


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
parser.add_argument("task_file",help="task file", default=None)
args = parser.parse_args()
model_file = args.model_file
taskFile = args.task_file

num_test = 1000
if taskFile is not None:
    num_test = 1
wandb.config.model_name = model_file
wandb.config.num_test = num_test

if not taskFile:
    env = DummyVecEnv([lambda: ProcessorEnv()])
else:
    env = DummyVecEnv([lambda: TestEnv(taskFile=taskFile)])

model = DQN.load(model_file)
model.set_env(env)

obs = env.reset()
done = False
episode_count = 0
taskID = -1
cumInstructions = 0
avgPowerPerInstr = 0
avgDelayPerTask = 0
skip = 0


ep_instr = []
ep_power = []
ep_power_per_instr = []
ep_delay_per_task = []
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
            avgPowerPerInstr += info[0]['power_consumed']
            delayPerTask = delayPerTask/len(info[0]['info'])
            ep_instr.append(totalInstr)
            ep_power.append(info[0]['power_consumed'])
            ep_power_per_instr.append(ep_power[-1]/ep_instr[-1])
            ep_delay_per_task.append(delayPerTask)
            wandb.log({'ep_instr':ep_instr[-1],
                        'ep_power':ep_power[-1],
                        'ep_power_per_instr':ep_power_per_instr[-1],
                        'ep_delay_per_task':ep_delay_per_task[-1]
                        })

        cumInstructions += totalInstr
        avgDelayPerTask += delayPerTask
        if num_test == 1:
            writeOutputFile('out.csv', info[0]['info'])
            env.env_method('graphShow', 'power')
            env.env_method('graphShow', 'temp')
        
avgPowerPerInstr = avgPowerPerInstr/(num_test - skip)
cumInstructions = cumInstructions/(num_test - skip)
avgDelayPerTask = avgDelayPerTask/(num_test - skip)
print("Mean Instruction Count per episode = \t" + str(cumInstructions))
print("avgPowerPerInstr = \t\t" + str(avgPowerPerInstr))
# print("cumInstructions = " + str(cumInstructions))
print("avgDelayPerTask = \t\t" + str(avgDelayPerTask))
print("skips = \t\t" + str(skip))
wandb.config.mean_ep_instr = np.mean(ep_instr)
wandb.config.mean_power = np.mean(ep_power)
wandb.config.mean_power_per_instr = np.mean(ep_power_per_instr)
wandb.config.mean_delay_per_task = np.mean(ep_delay_per_task)