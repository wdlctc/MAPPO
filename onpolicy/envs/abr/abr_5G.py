import numpy as np
from . import core as abrenv
from gym.spaces import Discrete
# -------------------------------------------------------------------------------
# Environment API
# -------------------------------------------------------------------------------


class Environment(object):
    """Abstract Environment interface.

    All concrete implementations of an environment should derive from this
    interface and implement the method stubs.
    """

    def seed(self, seed):
        raise NotImplementedError("Not implemented in Abstract Base class")

    def reset(self, config):
        r"""Reset the environment with a new config.

        Signals environment handlers to reset and restart the environment using
        a config dict.

        Args:
          config: dict, specifying the parameters of the environment to be
            generated.

        Returns:
          observation: A dict containing the full observation state.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def step(self, action):
        """Take one step in the game.

        Args:
          action: dict, mapping to an action taken by an agent.

        Returns:
          observation: dict, Containing full observation state.
          reward: float, Reward obtained from taking the action.
          done: bool, Whether the game is done.
          info: dict, Optional debugging information.

        Raises:
          AssertionError: When an illegal action is provided.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")

    def close(self):
        """Take one step in the game.

        Raises:
          AssertionError: abnormal close.
        """
        raise NotImplementedError("Not implemented in Abstract Base class")
    
    
class abrEnv(Environment):
    """RL interface to a abr environment.

    ```python

    environment = rl_env.make()
    config = { 'players': 5 }
    observation = environment.reset(config)
    while not done:
        # Agent takes action
        action =  ...
        # Environment take a step
        observation, reward, done, info = environment.step(action)
    ```
    """

    def __init__(self, args, seed):
        self._seed = seed
        self.num_agents = args.num_agents

        self.config = {
            "seed": self._seed,
            "DEFAULT_QUALITY": 1, 
            "S_INFO": 6,
            "S_LEN": 8,
            "A_DIM": 6,
            "VIDEO_BIT_RATE": np.array([300., 750., 1200., 1850., 2850., 4300.]),
            "BUFFER_NORM_FACTOR": 10.0,
            "M_IN_K": 1000.0,
            "CHUNK_TIL_VIDEO_END_CAP": 48.0,
            "REBUF_PENALTY": 4.3,  # 1 sec rebuffering -> 3 Mbps
            "SMOOTH_PENALTY": 1,
            "DEFAULT_QUALITY": 1  # default video quality without agent
        }

        self.net_env = abrenv.Environment(self.config)

        self.last_bit_rate = self.config["DEFAULT_QUALITY"]
        self.buffer_size = 0
        self.state = np.zeros(self.config["S_INFO"] * self.config["S_LEN"])

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        if self.num_agents == 1:
            self.action_space.append(Discrete(self.num_moves()))
            self.observation_space.append(
                [self.vectorized_observation_shape()[0]])
            self.share_observation_space.append(
                [self.vectorized_share_observation_shape()[0]])
        else:
            raise NotImplementedError("Not implemented for multiple user")

        print(self.action_space, self.observation_space)

    def reset(self):
        self.time_stamp = 0
        self.last_bit_rate = self.config["DEFAULT_QUALITY"]
        self.buffer_size = 0.
        self.state = np.zeros((self.config["S_INFO"], self.config["S_LEN"]))
        bit_rate = self.last_bit_rate
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = self.config["VIDEO_BIT_RATE"][bit_rate] / \
            float(np.max(self.config["VIDEO_BIT_RATE"]))  # last quality
        state[1, -1] = self.buffer_size / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / self.config["M_IN_K"]  # kilo byte / ms
        state[3, -1] = float(delay) / self.config["M_IN_K"] / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[4, :self.config["A_DIM"]] = np.array(
            next_video_chunk_sizes) / self.config["M_IN_K"] / self.config["M_IN_K"]  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  self.config["BUFFER_NORM_FACTOR"]) / float(self.config["CHUNK_TIL_VIDEO_END_CAP"])
        self.state = state
        obs = state.flatten()

        return obs

    def step(self, action):
        bit_rate = int(action[0])
        delay, sleep_time, self.buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
            self.net_env.get_video_chunk(bit_rate)
            
        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms
        
        # reward is video quality - rebuffer penalty - smooth penalty
        reward = self.config["VIDEO_BIT_RATE"][bit_rate] / self.config["M_IN_K"] \
                 - self.config["REBUF_PENALTY"] * rebuf \
                 - self.config["SMOOTH_PENALTY"] * np.abs(self.config["VIDEO_BIT_RATE"][bit_rate] -
                                        self.config["VIDEO_BIT_RATE"][self.last_bit_rate]) / self.config["M_IN_K"]
        self.last_bit_rate = bit_rate
        state = np.roll(self.state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = self.config["VIDEO_BIT_RATE"][bit_rate] / \
            float(np.max(self.config["VIDEO_BIT_RATE"]))  # last quality
        state[1, -1] = self.buffer_size / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / self.config["M_IN_K"]  # kilo byte / ms
        state[3, -1] = float(delay) / self.config["M_IN_K"] / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[4, :self.config["A_DIM"]] = np.array(
            next_video_chunk_sizes) / self.config["M_IN_K"] / self.config["M_IN_K"]  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  self.config["BUFFER_NORM_FACTOR"]) / float(self.config["CHUNK_TIL_VIDEO_END_CAP"])
        self.state = state
        
        obs = state.flatten()
        done = np.array([end_of_video] * self.num_agents)
        info = {'bitrate': self.config["VIDEO_BIT_RATE"][bit_rate], 'rebuffer': rebuf}

        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def num_moves(self):
        """Returns the total number of moves in this game (legal or not).

        Returns:
          Integer, number of moves.
        """
        return self.config["A_DIM"]

    def vectorized_observation_shape(self):
        """Returns the total number of moves in this game (legal or not).

        Returns:
          Integer, number of moves.
        """
        return [self.config["S_INFO"] * self.config["S_LEN"]]

    def vectorized_share_observation_shape(self):
        """Returns the total number of moves in this game (legal or not).

        Returns:
          Integer, number of moves.
        """
        return [self.config["S_INFO"] * self.config["S_LEN"]]

    def close(self):
        pass