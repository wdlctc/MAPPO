import os
import numpy as np
from . import load_trace
import copy

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
VIDEO_SIZE_FILE = os.path.dirname(os.path.abspath(__file__)) + '/envivio/video_size_'

# LEO SETTINGS
HANDOVER_DELAY = 0.2  # sec
HANDOVER_WEIGHT = 1
SCALE_VIDEO_SIZE_FOR_TEST = 20
SCALE_VIDEO_LEN_FOR_TEST = 2

class Environment:
    def __init__(self, params):
        
        random_seed = params['seed']
        np.random.seed(random_seed)
        self.num_agents = params['num_agents']

        self.all_cooked_time = params['all_cooked_time']
        self.all_cooked_bw = params['all_cooked_bw']
        
        self.all_cooked_remain = [] # FIXME: meaning?
        for trace_idx in range(len(self.all_cooked_bw)):
            self.all_cooked_remain.append({})
            for sat_id, sat_bw in self.all_cooked_bw[trace_idx].items():
                self.all_cooked_remain[trace_idx][sat_id] = []
                for index in range(len(sat_bw)):
                    count = 0
                    while index + count < len(sat_bw) and sat_bw[index] != 0:
                        count += 1
                    self.all_cooked_remain[trace_idx][sat_id].append(count)

        # pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]
        self.cooked_remain = self.all_cooked_remain[self.trace_idx]

        self.connection = {}
        for sat_id, sat_bw in self.cooked_bw.items():
            self.connection[sat_id] = [-1 for _ in range(len(sat_bw))]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        
        # begin with time 1, init time 0; set: -1(previous), get/calc: no -1(current)
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = [self.mahimahi_start_ptr for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_start_ptr - 1] for _ in range(self.num_agents)]

        self.num_of_user_sat = {}

        # connect the satellite that has the best performance at first
        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent] - 1)
            self.cur_sat_id.append(cur_sat_id)
            self.connection[cur_sat_id][self.mahimahi_ptr[agent] - 1] = agent
            self.update_sat_info(cur_sat_id, self.mahimahi_ptr[agent], 1)
 
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.video_chunk_remain = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.next_sat_bandwidth = [[] for _ in range(self.num_agents)] # FIXME: no need?
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]
        self.next_sat_user_nums = [[] for _ in range(self.num_agents)]
       
        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def set_satellite(self, agent, ho=0):
        sat_id = self.next_sat_id[agent]

        if ho == 1:
            self.connection[sat_id][self.mahimahi_ptr[agent]] = agent
            if sat_id == self.cur_sat_id[agent]:
                return
            else:
                self.update_sat_info(sat_id, self.mahimahi_ptr[agent], 1)
                self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], -1)
                self.cur_sat_id[agent] = sat_id
                self.delay[agent] = HANDOVER_DELAY
                return sat_id

    def new_initial_state(self):
        return None

    def step_ahead(self, agent):
        self.connection[self.cur_sat_id[agent]][self.mahimahi_ptr[agent]] = agent
        self.mahimahi_ptr[agent] += 1

    def get_video_chunk(self, quality, agent):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter[agent]]
        
        # use the delivery opportunity in mahimahi
        delay = self.delay[agent]  # in ms
        self.delay[agent] = 0
        video_chunk_counter_sent = 0  # in bytes
       
        while True:  # download video chunk over mahimahi
            # see how many users are sharing sat's bw
            if self.get_num_of_user_sat(self.cur_sat_id[agent]) == 0:
                throughput = self.cooked_bw[self.cur_sat_id[agent]][self.mahimahi_ptr[agent]] \
                             * B_IN_MB / BITS_IN_BYTE
            else: # divide by sharing number
                throughput = self.cooked_bw[self.cur_sat_id[agent]][self.mahimahi_ptr[agent]] \
                             * B_IN_MB / BITS_IN_BYTE / self.get_num_of_user_sat(self.cur_sat_id[agent])
                
            if throughput == 0.0: # no way to download video
                # Do the forced handover
                # Connect the satellite that has the best performance at first
                cur_sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
                # add user to best sat, and remove it from current sat
                self.update_sat_info(cur_sat_id, self.mahimahi_ptr[agent], 1)
                self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], -1)

                self.connection[cur_sat_id][self.mahimahi_ptr[agent]] = agent
                self.cur_sat_id[agent] = cur_sat_id
                delay += HANDOVER_DELAY

            duration = self.cooked_time[self.mahimahi_ptr[agent]] \
                       - self.last_mahimahi_time[agent]
	    
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time[agent] += fractional_time
                # assert(self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr])
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr[agent]]
            # self.mahimahi_ptr[agent] += 1
            self.step_ahead(agent)

            if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[agent]]):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr[agent] = 1
                self.last_mahimahi_time[agent] = 0
                self.end_of_video[agent] = True

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
                duration = self.cooked_time[self.mahimahi_ptr[agent]] \
                           - self.last_mahimahi_time[agent]
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time[agent] += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time[agent] = self.cooked_time[self.mahimahi_ptr[agent]]
                # self.mahimahi_ptr[agent] += 1
                self.step_ahead(agent)
                
                if self.mahimahi_ptr[agent] >= len(self.cooked_bw[self.cur_sat_id[agent]]):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr[agent] = 1
                    self.last_mahimahi_time[agent] = 0
 
        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size[agent]

        self.video_chunk_counter[agent] += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter[agent]
        # TODO: bookmark 2
        # get info for next choice based on previous time stamp
        self.next_sat_bandwidth[agent], self.next_sat_id[agent] = self.get_better_bw_id(agent, self.mahimahi_ptr[agent] - 1)

        if self.video_chunk_counter[agent] >= TOTAL_VIDEO_CHUNCK:
            self.end_of_video[agent] = True
            self.buffer_size[agent] = 0
            self.video_chunk_counter[agent] = 0

            self.cur_sat_id[agent] = -1


            # # pick a random trace file
            # self.trace_idx = np.random.randint(len(self.all_cooked_time))
            # self.cooked_time = self.all_cooked_time[self.trace_idx]
            # self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # # randomize the start point of the video
            # # note: trace file starts with time 0
            # self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            # self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

            # # Refresh satellite info
            # self.cur_sat_id = self.get_best_sat_id()
            # self.available_sat_list = self.get_available_sats_id()

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter[agent]])
    
        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            self.end_of_video[agent], \
            video_chunk_remain, \
            self.next_sat_bandwidth[agent]
           
    def reset(self):
        
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.video_chunk_remain = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.next_sat_bandwidth = [[] for _ in range(self.num_agents)]
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]
        
        while True:
            # pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]
            
            for sat_id, sat_bw in self.cooked_bw.items():
                length = len(sat_bw)
                break
            for i in range(length):
                result = []
                for sat_id, sat_bw in self.cooked_bw.items():
                    if sat_bw[i] > 0:
                        result.append(sat_bw[i])
                if len(result) < self.num_agents:
                    break
            if len(result) < self.num_agents:
                continue
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0     
            break
                

        self.mahimahi_ptr = [1 for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_start_ptr - 1] for _ in range(self.num_agents)]
        
        self.connection = {}
        for sat_id, sat_bw in self.cooked_bw.items():
            self.connection[sat_id] = [-1 for _ in range(len(sat_bw))]

        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent] - 1)
            self.cur_sat_id.append(cur_sat_id)
            self.connection[cur_sat_id][self.mahimahi_ptr[agent] - 1] = agent
        
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

    def get_best_sat_id(self, agent, mahimahi_ptr=None):
        best_sat_id = None
        best_sat_bw = 0

        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]

        for sat_id, sat_bw in self.cooked_bw.items():
            if best_sat_bw < sat_bw[mahimahi_ptr]:
                if self.connection[sat_id][mahimahi_ptr] == -1 or self.connection[sat_id][mahimahi_ptr] == agent:
                    best_sat_id = sat_id
                    best_sat_bw = sat_bw[mahimahi_ptr]
        
        if best_sat_id == None:
            best_sat_id = self.cur_sat_id[agent]
        return best_sat_id


    def get_better_bw_id(self, agent, mahimahi_ptr=None):
        if mahimahi_ptr is None:
            mahimahi_ptr = self.mahimahi_ptr[agent]
        
        # better bandwidth and according sat id (better than cur_sat)
        better_sat_bandwidth, better_sat_id = [], []

        # add cur_id to better list first
        sat_bw = self.cooked_bw[self.cur_sat_id[agent]] # bandwidth of cur_sat over all time
        # average over 5 timestamp
        bw_list_of_5timestamp = []
        for i in range(5, 0, -1):
            if mahimahi_ptr - i >= 0:
                num_of_user = self.get_num_of_user_sat(self.cur_sat_id[agent])
                if num_of_user == 0:
                    bw_list_of_5timestamp.append(sat_bw[mahimahi_ptr - i])
                else:
                    bw_list_of_5timestamp.append(sat_bw[mahimahi_ptr - i] / num_of_user)
        avg_bw_of_5timestamp = sum(bw_list_of_5timestamp) / len(bw_list_of_5timestamp)
        better_sat_bandwidth.append(avg_bw_of_5timestamp), better_sat_id.append(self.cur_sat_id[agent])
        cur_bw_log = copy.copy(bw_list_of_5timestamp)

        # then go over all sats to find better bw and according id
        best_sat_id = None
        best_sat_bw = 0
        best_bw_log = []
        for sat_id, sat_bw in self.cooked_bw.items():
            if sat_id == self.cur_sat_id[agent]:
                continue
            # average bw over 5 timestamp
            bw_list_of_5timestamp = []
            for i in range(5, 0, -1):
                if mahimahi_ptr - i >= 0 and sat_bw[mahimahi_ptr - i] != 0:
                    num_of_user = self.get_num_of_user_sat(sat_id)
                    if num_of_user == 0:
                        bw_list_of_5timestamp.append(sat_bw[mahimahi_ptr - i])
                    else: # if add agent, share bw with already connected agents, so +1
                        bw_list_of_5timestamp.append(sat_bw[mahimahi_ptr - i] / (num_of_user + 1))
            if len(bw_list_of_5timestamp) == 0:
                continue
            avg_bw_of_5timestamp = sum(bw_list_of_5timestamp) / len(bw_list_of_5timestamp)
            if best_sat_bw < avg_bw_of_5timestamp:
                # if at time mahimahi_ptr, sat_id is not used by other agent, or already used by agent
                if self.connection[sat_id][mahimahi_ptr] == -1 or self.connection[sat_id][mahimahi_ptr] == agent:
                    best_sat_id = sat_id
                    best_sat_bw = avg_bw_of_5timestamp
                    best_bw_log = copy.copy(bw_list_of_5timestamp)

        if best_sat_id == None:
            best_sat_id = self.cur_sat_id[agent]
        # append best solution as better choice -> also next choice
        better_sat_bandwidth.append(best_sat_bw), better_sat_id.append(best_sat_id)
        # TODO: after uptime_list...
        return cur_bw_log, better_sat_bandwidth, better_sat_id, best_bw_log

    def update_sat_info(self, sat_id, mahimahi_ptr, variation):
        # update sat info
        if sat_id in self.num_of_user_sat.keys():
            self.num_of_user_sat[sat_id] += variation
        else:
            self.num_of_user_sat[sat_id] = variation

        assert self.num_of_user_sat[sat_id] >= 0

    def get_num_of_user_sat(self, sat_id):
        # update sat info
        if sat_id == "all":
            return self.num_of_user_sat
        if sat_id in self.num_of_user_sat.keys():
            return self.num_of_user_sat[sat_id]

        return 0