import numpy as np
# from termcolor import colored
from loguru import logger

class HistoryHandler:
    
    def __init__(self, history_config, obs_dims):
        self.obs_dims = obs_dims
        self.history = {}

        self.buffer_config = {}
        for obs_key, obs_num in history_config.items():
            if obs_key in self.buffer_config:
                self.buffer_config[obs_key] = max(self.buffer_config[obs_key], obs_num)
            else:
                self.buffer_config[obs_key] = obs_num
        
        for key in self.buffer_config.keys():
            print(f"Key: {key}, Value: {self.buffer_config[key]}")
            self.history[key] = np.zeros((1, self.buffer_config[key], obs_dims[key]))

        logger.info("History Handler Initialized")
        for key, value in self.buffer_config.items():
            logger.info(f"Key: {key}, Value: {value}")

    def reset(self, reset_ids):
        if len(reset_ids)==0:
            return
        assert set(self.buffer_config.keys()) == set(self.history.keys()), f"History keys mismatch\n{self.buffer_config.keys()}\n{self.history.keys()}"
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.

    def add(self, key: str, value: np.ndarray):
        assert key in self.history.keys(), f"Key {key} not found in history"
        val = self.history[key][:]
        self.history[key][:, 1:] = val[:, :-1]
        self.history[key][:, 0] = value[:]
        
    def query(self, key: str):
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key][:]