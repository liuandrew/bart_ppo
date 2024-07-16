from gymnasium.envs.registration import register
register(
    id='BartEnv-v0',
    entry_point='gym_bart.envs:BartEnv',
)
register(
    id='BartMetaEnv',
    entry_point='gym_bart.envs:BartMetaEnv',
)