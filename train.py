import os
import argparse
import csv
import gym
import numpy as np
import matplotlib.pyplot as plt

from gym_microprocessor.envs.microprocessor_env import ProcessorEnv

from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.deepq import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2, DQN
# from stable_baselines.ddpg import AdaptiveParamNoiseSpec
from stable_baselines import results_plotter

from stable_baselines.common.callbacks import BaseCallback
import wandb

wandb.init( project="gym-microprocessor")
wandb.config["more"] = "custom"

parser = argparse.ArgumentParser()
parser.add_argument("-c",help="continue from aborted model", action="store_true")
args = parser.parse_args()
resume = args.c
# os._exit()
best_mean_reward, n_steps = -np.inf, 0

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

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, env, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf
        self.env = env

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
        #   x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          y = self.env.get_attr('ep_rew')
          ep_len = self.env.get_attr('episode_lengths')
          y=y[0]
          ep_len = ep_len[0]
          ep_power_saved = self.env.get_attr('ep_power_saved')[0]
          ep_power_per_instr = self.env.get_attr('ep_power_per_instr')[0]
          ep_delay_per_task = self.env.get_attr('ep_delay_per_task')[0]
          ep_instr = self.env.get_attr('ep_instr')[0]
          if len(y) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              mean_episode_length = np.mean(ep_len[-100:])
              mean_delay_per_task = np.mean(ep_delay_per_task[-100:])
              mean_power_saved = np.mean(ep_power_saved[-100:])
              mean_power_per_instr = np.mean(ep_power_per_instr[-100:])
              mean_episode_instr =np.mean(ep_instr[-100:])
              wandb.log({"mean_reward": mean_reward,
                         "mean_episode_length": mean_episode_length, 
                         "mean_power_saved": mean_power_saved,
                         "mean_power_per_instr": mean_power_per_instr,
                         "mean_delay_per_task" : mean_delay_per_task,
                         "mean_episode_instr": mean_episode_instr})

              if self.verbose > 0:
                print("Num timesteps: {} \t Num Episodes: {}".format(self.num_timesteps,len(y)))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}\n\n".format(self.best_mean_reward, mean_reward))
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True



# Create log dir
log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

models_dir = "models/"
os.makedirs(models_dir, exist_ok=True)

model_name = 'ppo2_resetnew_1_expt8_CONT_resetnew'
learning_rate = float(1 * 1e-5)
time_steps = int(1e7)

wandb.config.model_name = model_name
wandb.config.learning_rate = learning_rate
wandb.config.time_steps = time_steps

# Create and wrap the environment
env =  DummyVecEnv([lambda: ProcessorEnv()])
# env = DummyVecEnv([lambda: ProcessorEnv(taskFile='data/dataset/CSV/0.csv')])

# np.random.seed(123)
# env.seed(123)
# print (env.get_attr('reward_range'))
# env.reward_range = env.get_attr('reward_range')
# env = Monitor(env, log_dir, allow_early_resets=True)

# Because we use parameter noise, we should use a MlpPolicy with layer normalization
if (resume):
    model = PPO2.load(models_dir + "ppo2_resetnew_noroundoff_1_expt8")
    model.set_env(env)
    print("RESUMED")
else:
    model = PPO2(MlpPolicy, env, verbose=0, learning_rate=learning_rate)
    print(float(1e-5)==0.00001)
# Create the callback: check every 1000 steps
callback = SaveOnBestTrainingRewardCallback(env = env,check_freq=1000, log_dir=log_dir)
# Train the agent


try:
    model.learn(total_timesteps=int(time_steps), callback= callback)
    model.save(models_dir + model_name)
except KeyboardInterrupt:
    model.save(models_dir + model_name + "_abort")
finally:
    mean_episode_reward = env.get_attr('mean_episode_reward')
    print (mean_episode_reward)
    plt.plot(mean_episode_reward[0],'r.-', label="Mean Episode Reward(100)")
    mean_episode_length = env.get_attr('mean_episode_length')
    print (mean_episode_length)
    plt.plot(mean_episode_length[0],'g.-', label="Mean Episode Length(100)")
    plt.legend()
    plt.show()


t_steps = [i for i,k in enumerate(mean_episode_reward[0])]
tp = [(np.array(t_steps), np.array(mean_episode_reward[0]))]
print(tp)
results_plotter.plot_curves(tp,'timesteps',"TITLE")
plt.show()

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
        env.env_method('graphShow','power')
        env.env_method('graphShow','temp')

