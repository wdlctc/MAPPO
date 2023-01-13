import os
import itertools

from scipy.optimize import minimize, LinearConstraint
from statsmodels.tsa.api import ExponentialSmoothing
import pandas as pd
import numpy as np
import copy
from .satellite import Satellite
from .user import User
from .constants import EPSILON, MPC_FUTURE_CHUNK_COUNT, QUALITY_FACTOR, REBUF_PENALTY, SMOOTH_PENALTY, \
    MPC_PAST_CHUNK_COUNT, HO_NUM, TOTAL_VIDEO_CHUNKS, CHUNK_TIL_VIDEO_END_CAP, DEFAULT_QUALITY

VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
M_IN_K = 1000.0

MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
PAST_LEN = 8
TOTAL_VIDEO_CHUNCK = 48
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
    + '/dataset/envivio/video_size_'

# LEO SETTINGS
HANDOVER_DELAY = 0.2  # sec
HANDOVER_WEIGHT = 1
SCALE_VIDEO_SIZE_FOR_TEST = 20
SCALE_VIDEO_LEN_FOR_TEST = 2

SAT_STRATEGY = "resource-fair"
# SAT_STRATEGY = "ratio-based"

SNR_MIN = 70

BUF_RATIO = 0.7
class Environment:
    def __init__(self, params):
        
        random_seed = params['seed']
        np.random.seed(random_seed)
        self.num_agents = params['num_agents']

        self.all_cooked_time = params['all_cooked_time']
        self.all_cooked_bw = params['all_cooked_bw']
        

        # pick a random trace file
        self.trace_idx = 5
        #self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.last_quality = [DEFAULT_QUALITY for _ in range(self.num_agents)]
        

        # randomize the start point of the trace
        # note: trace file starts with time 0
        
        # begin with time 1, init time 0; set: -1(previous), get/calc: no -1(current)
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = [self.mahimahi_start_ptr for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_start_ptr - 1] for _ in range(self.num_agents)]
        # FIXME: KJ: self.mahimahi_start_ptr - 1
        
        # Centralization
        self.user_qoe_log = [{} for _ in range(self.num_agents)]
        self.num_sat_info = {}
        self.cur_satellite = {}
        # sat info
        for sat_id, sat_bw in self.cooked_bw.items():
            self.num_sat_info[sat_id] = [0 for _ in range(len(sat_bw))]
            self.cur_satellite[sat_id] = Satellite(sat_id, sat_bw, SAT_STRATEGY)
        # user info
        self.cur_user = []
        for agent_id in range(self.num_agents):
            self.cur_user.append(User(agent_id, SNR_MIN))

        # connect the satellite that has the best performance at first
        self.cur_sat_id = []
        self.prev_sat_id = [None for _ in range(self.num_agents)]
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent)
            self.cur_sat_id.append(cur_sat_id)
            self.update_sat_info(cur_sat_id, self.mahimahi_ptr[agent], agent, 1)
 
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]
       
        # FIXME: don't know if these variables are useful
        self.bit_rate = None
        self.download_bw = [[] for _ in range(self.num_agents)]
        self.past_download_ests = [[] for _ in range(self.num_agents)]
        self.past_download_bw_errors = [[] for _ in range(self.num_agents)]
        self.past_bw_ests = [{} for _ in range(self.num_agents)]
        self.past_bw_errors = [{} for _ in range(self.num_agents)]
        
        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))
        
        self.last_delay = [MPC_PAST_CHUNK_COUNT for _ in range(self.num_agents)]
        self.unexpected_change = True

    def get_video_chunk(self, quality, agent, model_type):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        if model_type is not None and (agent == 0 or self.unexpected_change) and self.end_of_video[agent] is not True:
            exit(1)

        
        # update noise of agent SNR
        self.cur_user[agent].update_snr_noise()
        
        video_chunk_size = self.video_size[quality][self.video_chunk_counter[agent]]
        
        # use the delivery opportunity in mahimahi
        delay = self.delay[agent]  # in ms
        self.delay[agent] = 0
        video_chunk_counter_sent = 0  # in bytes
        end_of_network = False

        # Do All users' handover
        self.last_quality[agent] = quality

        while True:  # download video chunk over mahimahi
            throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],\
                        self.mahimahi_ptr[agent]) * B_IN_MB / BITS_IN_BYTE
            if throughput == 0.0:
                # Connect the satellite that has the best serving time
                sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
                # add user to best sat, and remove it from current sat
                self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], agent, -1)
                self.update_sat_info(sat_id, self.mahimahi_ptr[agent], agent, 1)
                self.switch_sat(agent, sat_id)
                delay += HANDOVER_DELAY
                self.download_bw[agent] = []
                self.unexpected_change = True
                throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],\
                        self.mahimahi_ptr[agent]) * B_IN_MB / BITS_IN_BYTE
                assert throughput != 0

            duration = self.cooked_time[self.mahimahi_ptr[agent]] \
                       - self.last_mahimahi_time[agent]
	    
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time[agent] += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr[agent]]
            
            self.mahimahi_ptr[agent] += 1

            if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[agent]]):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr[agent] = 1
                self.last_mahimahi_time[agent] = 0
                self.end_of_video[agent] = True
                end_of_network = True
                break
        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

	    # add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size[agent], 0.0)

        # update the buffer
        self.buffer_size[agent] = np.maximum(self.buffer_size[agent] - delay, 0.0)

        # add in the new chunk
        self.buffer_size[agent] += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size[agent] > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size[agent] - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size[agent] -= sleep_time

            while True:
                if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[agent]]):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr[agent] = 1
                    self.last_mahimahi_time[agent] = 0
                    self.end_of_video[agent] = True
                    end_of_network = True
                    break
                
                duration = self.cooked_time[self.mahimahi_ptr[agent]] \
                           - self.last_mahimahi_time[agent]
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time[agent] += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr[agent]]
                self.mahimahi_ptr[agent] += 1
                
                if throughput == 0.0:
                    # Connect the satellite that has the best serving time
                    sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
                    # add user to best sat, and remove it from current sat
                    self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], agent, -1)
                    self.update_sat_info(sat_id, self.mahimahi_ptr[agent], agent, 1)
                    self.switch_sat(agent, sat_id)
                    delay += HANDOVER_DELAY
                    throughput = self.cur_satellite[self.cur_sat_id[agent]].data_rate(self.cur_user[agent],\
                            self.mahimahi_ptr[agent]) * B_IN_MB / BITS_IN_BYTE
 
        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size[agent]

        self.video_chunk_counter[agent] += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.video_chunk_counter[agent]

        # get info for next choice based on previous time stamp
        # these returned list has 2 elements: [0] for current, and [1] for best choice
        better_sat_bw, better_sat_id, better_sat_bw_log, \
            connected_time = self.get_better_bw_id(agent, self.mahimahi_ptr[agent] - 1)
        self.next_sat_id[agent] = better_sat_id[1]
        
        if self.video_chunk_counter[agent] >= TOTAL_VIDEO_CHUNKS or end_of_network:
            self.end_of_video[agent] = True
            self.buffer_size[agent] = 0
            self.video_chunk_counter[agent] = 0
            self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], agent, -1)
        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter[agent]])
    
        self.download_bw[agent].append(float(video_chunk_size) / delay / M_IN_K * BITS_IN_BYTE)
        # num of users
        # FIXME: whether ptr - 1
        cur_sat_user_num = len(self.cur_satellite[self.cur_sat_id[agent]].get_ue_list(self.mahimahi_ptr[agent]))
        next_sat_user_num = len(self.cur_satellite[self.next_sat_id[agent]].get_ue_list(self.mahimahi_ptr[agent]))
        MPC_PAST_CHUNK_COUNT = round(delay / M_IN_K)
        
        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            self.end_of_video[agent], \
            video_chunk_remain, \
            better_sat_bw,  better_sat_bw_log, \
            cur_sat_user_num, next_sat_user_num, \
            connected_time
           
    def reset(self):
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]
        self.download_bw = [[] for _ in range(self.num_agents)]
        self.cur_satellite = {}

        while True:
            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            sat_bw = list(self.cooked_bw.values())[0]
            length = len(sat_bw)
            for i in range(length):
                result = []
                for sat_id, sat_bw in self.cooked_bw.items():
                    if sat_bw[i] > 0:
                        result.append(sat_bw[i])
                if len(result) < self.num_agents:
                    break
            if len(result) < self.num_agents: # find a valid trace file
                continue
            break
        
        for sat_id, sat_bw in self.cooked_bw.items():
            self.num_sat_info[sat_id] = [0 for _ in range(len(sat_bw))]
            self.cur_satellite[sat_id] = Satellite(sat_id, sat_bw, SAT_STRATEGY)

        self.cur_user = []
        for agent_id in range(self.num_agents):
            self.cur_user.append(User(agent_id, SNR_MIN))
        self.mahimahi_ptr = [1 for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_start_ptr - 1] for _ in range(self.num_agents)]
        
        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
            self.cur_sat_id.append(cur_sat_id)
            self.update_sat_info(cur_sat_id, self.mahimahi_ptr[agent] - 1, agent, 1)
        self.last_delay = [MPC_PAST_CHUNK_COUNT for _ in range(self.num_agents)]
        
    def check_end(self):
        for agent in range(self.num_agents):
            if not self.end_of_video[agent]:
                return False
        return True
     
    def get_first_agent(self):
        user = -1
        
        for agent in range(self.num_agents):
            if not self.end_of_video[agent]:
                if user == -1:
                    user = agent
                else:
                    if self.last_mahimahi_time[agent] < self.last_mahimahi_time[user]:
                        user = agent
                        
        return user

    def get_average_bw(self, sat_id, mahimahi_ptr, smoothness=5):
        sat_bw = self.cooked_bw[sat_id]
        bw_list = []
        for i in range(smoothness):
            if mahimahi_ptr - i >= 0: # and sat_bw[mahimahi_ptr-i] != 0:
                num_of_user = self.get_num_of_user_sat(sat_id, mahimahi_ptr)
                if num_of_user == 0:
                    bw_list.append(sat_bw[mahimahi_ptr - i])
                else: # if add agent, share bw with already connected agents, so +1
                    bw_list.append(sat_bw[mahimahi_ptr - i] / (num_of_user))
            else:
                bw_list.append(0)
        return bw_list, sum(bw_list) / len(bw_list)
    
    def get_all_average_bw(self, mahimahi_ptr, agent):
        all_info_list = []
        for sat_id, sat_bw in self.cooked_bw.items():
            bw_list, bw = self.get_average_bw(sat_id, mahimahi_ptr, \
                smoothness=MPC_PAST_CHUNK_COUNT)
            all_info_list.append({
                'bw': bw,
                'sat_id': sat_id,
                'bw_list': bw_list
            })
        return all_info_list

    def get_best_bw(self, mahimahi_ptr, agent):
        all_info_list = self.get_all_average_bw(mahimahi_ptr, agent)
        best_sat_bw = 0
        best_sat_id = None
        best_sat_list = []
        for info in all_info_list:
            bw = info['bw']
            sat_id = info['sat_id']
            bw_list = info['bw_list']
            if best_sat_bw < bw:
                best_sat_id = sat_id
                best_sat_bw = bw
                best_sat_list = bw_list
        return best_sat_id, best_sat_bw, best_sat_list

    def get_better_bw_id(self, agent, mahimahi_ptr=None):
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]
        
        # better bandwidth and according sat id (better than cur_sat)
        better_sat_bw, better_sat_id, better_sat_bw_log = [], [], []

        # add cur_id to better list first
        # average bw over 5 timestamp
        bw_list, bw = self.get_average_bw(self.cur_sat_id[agent], mahimahi_ptr, \
            smoothness=MPC_PAST_CHUNK_COUNT)
        better_sat_bw.append(bw), better_sat_id.append(self.cur_sat_id[agent])
        better_sat_bw_log.append(bw_list)

        # then go over all sats to find better bw and according id
        best_sat_id, best_sat_bw, best_bw_log = self.get_best_bw(mahimahi_ptr, agent)
        if best_sat_id == None:
            best_sat_id = self.cur_sat_id[agent]
            best_sat_bw = 0
        better_sat_bw.append(best_sat_bw), better_sat_id.append(best_sat_id)
        better_sat_bw_log.append(best_bw_log)

        up_time_list = []
        up_time_list.append(self.get_uptime(self.cur_sat_id[agent], mahimahi_ptr))
        up_time_list.append(self.get_uptime(best_sat_id, mahimahi_ptr))
        
        return better_sat_bw, better_sat_id, better_sat_bw_log, up_time_list

    def get_best_sat_id(self, agent, mahimahi_ptr=None):
        best_sat_id = None
        best_sat_bw = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        for sat_id, sat_bw in self.cooked_bw.items():
            real_sat_bw = self.cur_satellite[sat_id].data_rate(self.cur_user[agent], mahimahi_ptr)
            if best_sat_bw < real_sat_bw:
                best_sat_id = sat_id
                best_sat_bw = real_sat_bw
            
        return best_sat_id

    def switch_sat(self, agent, cur_sat_id):
        pre_sat_id = self.cur_sat_id[agent]
        self.prev_sat_id[agent] = pre_sat_id
        self.cur_sat_id[agent] = cur_sat_id

    def get_uptime(self, sat_id, mahimahi_ptr):
        sat_bw = self.cooked_bw[sat_id]
        up_time = 0
        tmp_index = mahimahi_ptr
        tmp_sat_bw = sat_bw[tmp_index]
        while tmp_sat_bw != 0 and tmp_index >= 0:
            up_time += 1
            tmp_index -= 1
            tmp_sat_bw = sat_bw[tmp_index]
        return up_time

    def update_sat_info(self, sat_id, mahimahi_ptr, agent, variation):
        # update sat info
        if variation == 1:
            self.cur_satellite[sat_id].add_ue(agent, mahimahi_ptr)
        elif variation == -1:
            self.cur_satellite[sat_id].remove_ue(agent, mahimahi_ptr)


    def get_num_of_user_sat(self, sat_id, mahimahi_ptr):
        # update sat info
        if sat_id == "all":
            filtered_num_of_user_sat = {}
            for tmp_sat_id in self.cur_satellite.keys():
                if len(self.cur_satellite[tmp_sat_id].get_ue_list(mahimahi_ptr)) != 0:
                    filtered_num_of_user_sat[tmp_sat_id] = len(self.cur_satellite[tmp_sat_id].get_ue_list(mahimahi_ptr))
            return filtered_num_of_user_sat
        if sat_id in self.cur_satellite.keys():
            return len(self.cur_satellite[sat_id].get_ue_list(mahimahi_ptr))

    def set_satellite(self, agent, sat=0):
        sat_id = self.next_sat_id[agent]

        if sat == 1:
            if sat_id == self.cur_sat_id[agent]:
                return
            self.update_sat_info(sat_id, self.mahimahi_ptr[agent], agent, 1)
            self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], agent, -1)
            self.prev_sat_id[agent] = self.cur_sat_id[agent]
            self.cur_sat_id[agent] = sat_id
            self.download_bw[agent] = []
            self.delay[agent] = HANDOVER_DELAY
            return sat_id
