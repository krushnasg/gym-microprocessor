from gym.envs.registration import register

register(
    id='microprocessor-v0',
    entry_point='gym_microprocessor.envs:ProcessorEnv',
)