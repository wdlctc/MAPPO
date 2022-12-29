import numpy as np
from . import core_implicit as abrenv
from . import load_trace
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
            "S_INFO": 6+1+4, #FIXME: why kyoungjun set to 6+1+4
            "S_LEN": 8,
            "A_DIM": 6,
            "PAST_LEN": 8,
            "A_SAT" : 2,
            "SAT_DIM" : 2, # SAT_DIM = A_SAT
            "VIDEO_BIT_RATE": np.array([300., 750., 1200., 1850., 2850., 4300.]),
            "BUFFER_NORM_FACTOR": 10.0,
            "M_IN_K": 1000.0,
            "CHUNK_TIL_VIDEO_END_CAP": 48.0,
            "REBUF_PENALTY": 4.3,  # 1 sec rebuffering -> 3 Mbps
            "SMOOTH_PENALTY": 1,
            "DEFAULT_QUALITY": 1,
            "num_agents": self.num_agents,
        }

        self.is_handover = False
        all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()
        param = {
            'seed': self.config["seed"],
            'num_agents': self.config["num_agents"],
            'all_cooked_time': all_cooked_time,
            'all_cooked_bw': all_cooked_bw,
        }
        self.net_env = abrenv.Environment(param)

        self.last_bit_rate = self.config["DEFAULT_QUALITY"]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.state = [np.zeros((self.config["S_INFO"], self.config["S_LEN"]))for _ in range(self.num_agents)]
        self.sat_decision_log = [[] for _ in range(self.num_agents)]
        # FIXME: I remember LC wrote more codes after this? (compared to original pensive-PPO)
    
    def reset_agent(self, agent):
        bit_rate = self.config["DEFAULT_QUALITY"]
        delay, sleep_time, self.buffer_size[agent], rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, \
            _, next_sat_bw_logs, cur_sat_user_num, next_sat_user_nums, cur_sat_bw_logs, connected_time = \
            self.net_env.get_video_chunk(bit_rate, agent)
        state = np.roll(self.state[agent], -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = self.config["VIDEO_BIT_RATE"][bit_rate] / \
            float(np.max(self.config["VIDEO_BIT_RATE"]))  # last quality
        state[1, -1] = self.buffer_size[agent] / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / self.config["M_IN_K"]  # kilo byte / ms
        state[3, -1] = float(delay) / self.config["M_IN_K"] / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[4, :self.config["A_DIM"]] = np.array(
            next_video_chunk_sizes) / self.config["M_IN_K"] / self.config["M_IN_K"]  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  self.config["CHUNK_TIL_VIDEO_END_CAP"]) / float(self.config["CHUNK_TIL_VIDEO_END_CAP"])
        if len(next_sat_bw_logs) < self.config['PAST_LEN']:
            next_sat_bw_logs = [0] * (self.config['PAST_LEN'] - len(next_sat_bw_logs)) + next_sat_bw_logs
        state[6, :self.config['PAST_LEN']] = np.array(next_sat_bw_logs[:self.config['PAST_LEN']])

        if len(cur_sat_bw_logs) < self.config['PAST_LEN']:
            cur_sat_bw_logs = [0] * (self.config['PAST_LEN'] - len(cur_sat_bw_logs)) + cur_sat_bw_logs

        state[7, :self.config['PAST_LEN']] = np.array(cur_sat_bw_logs[:self.config['PAST_LEN']])
        if self.is_handover:
            state[8:9, 0:self.config['S_LEN']] = np.zeros((1, self.config['S_LEN']))
            state[9:10, 0:self.config['S_LEN']] = np.zeros((1, self.config['S_LEN']))

        state[8:9, -1] = np.array(cur_sat_user_num) / 10
        state[9:10, -1] = np.array(next_sat_user_nums) / 10

        state[10, :2] = [float(connected_time[0]) / self.config['BUFFER_NORM_FACTOR'] / 10, float(connected_time[1]) / self.config['BUFFER_NORM_FACTOR'] / 10]

        self.state[agent] = state
        
        return self.state[agent]

    def get_obs_from_state(self, agent):
        
        # TODO: this function didn't change 
        obs = np.zeros((1, 3+self.config["S_LEN"]*2+self.config["A_DIM"]))
        obs[0][0] = self.state[agent][0, -1]
        obs[0][1] = self.state[agent][1, -1]
        obs[0][2:2+self.config["S_LEN"]] = self.state[agent][2, :]
        obs[0][2+self.config["S_LEN"]:2+self.config["S_LEN"]*2] = self.state[agent][3, :]
        obs[0][2+self.config["S_LEN"]*2:2+self.config["S_LEN"]*2+self.config["A_DIM"]] = self.state[agent][4, :self.config["A_DIM"]]
        obs[0][2+self.config["S_LEN"]*2+self.config["A_DIM"]:3+self.config["S_LEN"]*2+self.config["A_DIM"]] = self.state[agent][5, -1]
        obs[0][3+self.config["S_LEN"]*2+self.config["A_DIM"]:3+self.config["S_LEN"]*2+self.config["A_DIM"]+self.config['SAT_DIM']] = self.state[agent][6, :self.config['SAT_DIM']]
        return obs

    def reset(self):

        self.net_env.reset()
        self.time_stamp = 0
        self.last_bit_rate = self.config["DEFAULT_QUALITY"]

        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.state = [np.zeros((self.config["S_INFO"] , self.config["S_LEN"]))for _ in range(self.num_agents)]

        for agent in range(self.num_agents):
            self.reset_agent(agent)
        
        agent = self.net_env.get_first_agent()

        agent_turn = np.zeros((1,self.num_agents), dtype=np.int).tolist()
        agent_turn[0][agent] = 1

        obs = self.get_obs_from_state(agent)
        share_obs = [self.get_obs_from_state(agent) for i in range(self.num_agents)]
        concat_obs = np.concatenate(share_obs, axis=1)
        share_obs = np.concatenate((concat_obs, agent_turn), axis=1)

        return obs

    def step(self, action):
        action = int(action[0])
        bit_rate = int(action) % self.config["A_DIM"]
        sat = int(action) // self.config["A_DIM"]
        agent = self.net_env.get_first_agent()

        self.net_env.set_satellite(agent, sat) # FIXME: what's the meaning?

        delay, sleep_time, self.buffer_size[agent], rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain, \
            next_sat_bw = \
            self.net_env.get_video_chunk(bit_rate, agent)
            
        self.time_stamp += delay  # in ms
        self.time_stamp += sleep_time  # in ms
        
        # reward is video quality - rebuffer penalty - smooth penalty
        reward = self.config["VIDEO_BIT_RATE"][bit_rate] / self.config["M_IN_K"] \
                 - self.config["REBUF_PENALTY"] * rebuf \
                 - self.config["SMOOTH_PENALTY"] * np.abs(self.config["VIDEO_BIT_RATE"][bit_rate] -
                                        self.config["VIDEO_BIT_RATE"][self.last_bit_rate]) / self.config["M_IN_K"]
        self.dones[agent] = end_of_video
        self.rewards[agent][0] = reward

        self.last_bit_rate = bit_rate
        state = np.roll(self.state[agent], -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = self.config["VIDEO_BIT_RATE"][bit_rate] / \
            float(np.max(self.config["VIDEO_BIT_RATE"]))  # last quality
        state[1, -1] = self.buffer_size[agent] / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[2, -1] = float(video_chunk_size) / \
            float(delay) / self.config["M_IN_K"]  # kilo byte / ms
        state[3, -1] = float(delay) / self.config["M_IN_K"] / self.config["BUFFER_NORM_FACTOR"]  # 10 sec
        state[4, :self.config["A_DIM"]] = np.array(
            next_video_chunk_sizes) / self.config["M_IN_K"] / self.config["M_IN_K"]  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain,
                                  self.config["CHUNK_TIL_VIDEO_END_CAP"]) / float(self.config["CHUNK_TIL_VIDEO_END_CAP"])
        state[6, :self.config["SAT_DIM"]] = np.array(next_sat_bw)
        self.state[agent] = state

        obs = self.get_obs_from_state(agent)
        
        done = self.dones
        info = {'bitrate': self.config["VIDEO_BIT_RATE"][bit_rate],
                'rebuffer': rebuf, 
                'time_stamp':self.time_stamp, 
                'reward':reward, 
                'agent': agent,
                'done': end_of_video}
        reward = self.rewards

        if end_of_video:
            state = np.zeros((self.config["S_INFO"], self.config["S_LEN"]))
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
        return self.config["A_DIM"] * 2

    def vectorized_observation_shape(self):
        """Returns the total number of moves in this game (legal or not).

        Returns:
          Integer, number of moves.
        """
        #return [3+self.config["S_LEN"]*2+self.config["A_DIM"]]
        return [3+self.config["S_LEN"]*2+self.config["A_DIM"]+self.config['SAT_DIM']] # new shape

    def vectorized_share_observation_shape(self):
        """Returns the total number of moves in this game (legal or not).

        Returns:
          Integer, number of moves.
        """
        return [3+self.config["S_LEN"]*2+self.config["A_DIM"]+self.config['SAT_DIM']]

    def close(self):
        pass