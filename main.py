import gym

from gym_microprocessor.envs.microprocessor_env import ProcessorEnv

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# The algorithms require a vectorized environment to run
env =  DummyVecEnv([lambda: ProcessorEnv()])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=2000)
# model.save("ppo2_microprocessor")
model = PPO2.load("ppo2_microprocessor")

obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()