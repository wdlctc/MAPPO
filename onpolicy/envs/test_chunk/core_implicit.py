import os
import numpy as np
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
VIDEO_SIZE_FILE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\
    + '/dataset/envivio/video_size_'

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
        

        # start from first trace
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # 2 variables relevant to time
        self.connection = {}
        self.num_of_user_sat = {}
        for sat_id, sat_bw in self.cooked_bw.items():
            self.connection[sat_id] = [-1 for _ in range(len(sat_bw))]
            self.num_of_user_sat[sat_id] = [0 for _ in range(len(sat_bw))]

        # randomize the start point of the trace
        # note: trace file starts with time 0
        
        # begin with time 1, init time 0; set: -1(previous), get/calc: no -1(current)
        self.mahimahi_start_ptr = 1
        self.mahimahi_ptr = [self.mahimahi_start_ptr for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_start_ptr - 1] for _ in range(self.num_agents)]

        # connect the satellite that has the best performance at first
        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent])
            self.cur_sat_id.append(cur_sat_id)
            self.connection[cur_sat_id][self.mahimahi_ptr[agent]] = agent
            self.update_sat_info(cur_sat_id, self.mahimahi_ptr[agent], 1)
 
        self.video_chunk_counter = [0 for _ in range(self.num_agents)]
        self.buffer_size = [0 for _ in range(self.num_agents)]
        self.video_chunk_counter_sent = [0 for _ in range(self.num_agents)]
        self.video_chunk_remain = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.next_sat_bandwidth = [[] for _ in range(self.num_agents)]
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]
       
        self.video_size = {}  # in bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    # def set_satellite(self, agent, ho=0):
    #     sat_id = self.next_sat_id[agent][ho]

    #     if ho == 1:
    #         self.connection[sat_id][self.mahimahi_ptr[agent]] = agent
    #         if sat_id == self.cur_sat_id[agent]:
    #             return
    #         else:
    #             self.update_sat_info(sat_id, 1)
    #             self.update_sat_info(self.cur_sat_id[agent], -1)
    #             self.cur_sat_id[agent] = sat_id
    #             self.delay[agent] = HANDOVER_DELAY
    #             return sat_id
            
    def set_satellite(self, agent, sat, id_list = None):
        if id_list == None:
            id_list = self.next_sat_id[agent]

        # Do not do any satellite switch
        sat_id = id_list[sat]
        self.connection[sat_id][self.mahimahi_ptr[agent]] = agent
        #self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], 1)

        if sat_id == self.cur_sat_id[agent]:
            return
        else: # handover accur
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
            #self.update_sat_info(self.cur_sat_id[agent], self.mahimahi_ptr[agent], 1)
            if self.get_num_of_user_sat(self.cur_sat_id[agent], self.mahimahi_ptr[agent]) == 0:
                throughput = self.cooked_bw[self.cur_sat_id[agent]][self.mahimahi_ptr[agent]] \
                             * B_IN_MB / BITS_IN_BYTE
            else: # divide by sharing number
                throughput = self.cooked_bw[self.cur_sat_id[agent]][self.mahimahi_ptr[agent]] \
                             * B_IN_MB / BITS_IN_BYTE / self.get_num_of_user_sat(self.cur_sat_id[agent], self.mahimahi_ptr[agent])
                
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

        # get info for next choice based on previous time stamp
        # these returned list has 2 elements: [0] for current, and [1] for best choice
        better_sat_bw, better_sat_id, better_sat_bw_log, \
            connected_time = self.get_better_bw_id(agent, self.mahimahi_ptr[agent] - 1)
        self.next_sat_id[agent] = better_sat_id
        
        if self.video_chunk_counter[agent] >= TOTAL_VIDEO_CHUNCK:
            self.end_of_video[agent] = True
            self.buffer_size[agent] = 0
            self.video_chunk_counter[agent] = 0
            self.cur_sat_id[agent] = -1
        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter[agent]])
    
        # num of users
        # should be ptr - 1: according to get_better_bw_id()
        cur_sat_user_num = self.get_num_of_user_sat(better_sat_id[0], self.mahimahi_ptr[agent] - 1)
        next_sat_user_num = self.get_num_of_user_sat(better_sat_id[1], self.mahimahi_ptr[agent] - 1)

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
        self.video_chunk_remain = [0 for _ in range(self.num_agents)]
        self.end_of_video = [False for _ in range(self.num_agents)]
        self.next_video_chunk_sizes = [[] for _ in range(self.num_agents)]
        self.next_sat_bandwidth = [[] for _ in range(self.num_agents)]
        self.next_sat_id = [[] for _ in range(self.num_agents)]
        self.delay = [0 for _ in range(self.num_agents)]

        self.trace_idx += 1
        if self.trace_idx >= len(self.all_cooked_time):
            self.trace_idx = 0
        #print("trace index: ", self.trace_idx)
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_ptr = [1 for _ in range(self.num_agents)]
        self.last_mahimahi_time = [self.cooked_time[self.mahimahi_start_ptr - 1] for _ in range(self.num_agents)]
        
        self.connection = {}
        self.num_of_user_sat = {}
        for sat_id, sat_bw in self.cooked_bw.items():
            self.connection[sat_id] = [-1 for _ in range(len(sat_bw))]
            self.num_of_user_sat[sat_id] = [0 for _ in range(len(sat_bw))]

        self.cur_sat_id = []
        for agent in range(self.num_agents):
            cur_sat_id = self.get_best_sat_id(agent, self.mahimahi_ptr[agent] - 1)
            self.cur_sat_id.append(cur_sat_id)
            self.connection[cur_sat_id][self.mahimahi_ptr[agent] - 1] = agent
            self.update_sat_info(cur_sat_id, self.mahimahi_ptr[agent] - 1, variation=1)
        
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

    def get_average_bw(self, sat_id, mahimahi_ptr, smoothness=5, agent=None):
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
            bw_list, bw = self.get_average_bw(sat_id, mahimahi_ptr, agent=agent)
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
                if self.connection[sat_id][mahimahi_ptr + 1] == -1:
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
        bw_list, bw = self.get_average_bw(self.cur_sat_id[agent], mahimahi_ptr)
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
            if sat_bw[mahimahi_ptr] == 0:
                continue
            bw_list, bw = self.get_average_bw(sat_id, mahimahi_ptr, smoothness=PAST_LEN)
            if best_sat_bw < bw:
                # I think this judgement is unecessary, 'cause in get_avg_bw(), we already considered user_num
                #if self.connection[sat_id][mahimahi_ptr] == -1 or self.connection[sat_id][mahimahi_ptr] == agent:
                best_sat_id = sat_id
                best_sat_bw = bw
        
        if best_sat_id == None:
            best_sat_id = self.cur_sat_id[agent]
        return best_sat_id

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

    def update_sat_info(self, sat_id, mahimahi_ptr, variation):
        # update sat info
        if sat_id in self.num_of_user_sat.keys():
            # next time's user nums is equal to current time by default
            # if variation != 0:
            #     self.num_of_user_sat[sat_id][mahimahi_ptr] += variation
            # else:
            #     self.num_of_user_sat[sat_id][mahimahi_ptr] = \
            #         self.num_of_user_sat[sat_id][mahimahi_ptr - 1]
            for i in range(mahimahi_ptr, len(self.cooked_bw)):
                self.num_of_user_sat[sat_id][i] += variation
        else:
            # self.num_of_user_sat[sat_id][mahimahi_ptr] = variation
            for i in range(mahimahi_ptr, len(self.cooked_bw)):
                self.num_of_user_sat[sat_id][i] = variation

        assert self.num_of_user_sat[sat_id][mahimahi_ptr] >= 0

    def get_num_of_user_sat(self, sat_id, mahimahi_ptr):
        # update sat info
        if sat_id == "all":
            return [self.num_of_user_sat[i][mahimahi_ptr] for i in self.num_of_user_sat.keys()]
        if sat_id in self.num_of_user_sat.keys():
            return self.num_of_user_sat[sat_id][mahimahi_ptr]

        return 0
