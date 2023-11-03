import gymnasium
import numpy as np


class ObservationWrapper(gymnasium.Wrapper):
    def __init__(self, env, forecast_length):
        super().__init__(env)

        self.forecast_length = forecast_length
        original_observation_space_length = self.observation_space.shape[0]
        self.observation_space = gymnasium.spaces.Box(shape=(original_observation_space_length - forecast_length,),
                                                      low=-np.inf,
                                                      high=np.inf, dtype=np.float32)

    def reset(self, seed: int = 42, options=None):
        obs, info = self.env.reset()
        return self.convert_observation(obs), info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        return self.convert_observation(obs), reward, done, trunc, info

    def convert_observation(self, obs):
        load_forecast = obs[1: self.forecast_length + 1]
        generation_forecast = obs[self.forecast_length + 1: 2 * self.forecast_length + 1]
        price_forecast = obs[2 * self.forecast_length + 1: 3 * self.forecast_length + 1]
        soc = obs[0]
        return np.concatenate(([soc],
                               load_forecast - generation_forecast,
                               price_forecast),
                              axis=0)
