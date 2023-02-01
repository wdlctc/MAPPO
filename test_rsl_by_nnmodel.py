import os
# preparation
users = [3,4,5]
from3_prefix = "/home/t-chengluo/xuyi-MAPPO/LC-MAPPO/results/abr_sat/time/ppo/2000mstest_hz128/run4/modelsnn_model_ep_"
from4_prefix = "/home/t-chengluo/xuyi-MAPPO/LC-MAPPO/results/abr_sat/time/ppo/2000mstest_hz128/run5/modelsnn_model_ep_"
from5_prefix = "/home/t-chengluo/xuyi-MAPPO/LC-MAPPO/results/abr_sat/time/ppo/2000mstest_hz128/run6/modelsnn_model_ep_"
from3_number = [3.4e6, 3.5e6, 5.2e6, 5.6e6, 5.8e6, 5.9e6, 6e6, 6.1e6, \
    6.6e6, 7.1e6, 7.3e6, 7.4e6, 7.5e6, 7.9e6, 8.7e6, 9e6, 9.1e6, 9.4e6, 9.5e6, 9.6e6]
from4_number = [1.4e6, 1.5e6, 1.6e6, 2.1e6, 2.9e6, 3e6, 3.1e6, 3.2e6, \
    3.4e6, 3.5e6, 4.9e6, 5.2e6, 6.2e6, 6.3e6, 6.4e6, 6.6e6, 6.7e6, 6.9e6, \
    7.1e6, 7.2e6, 7.6e6, 7.9e6, 8e6, 8.8e6, 9.2e6, 9.8e6]
from5_number = [1.7e6, 1.9e6, 2.5e6, 2.6e6, 2.9e6, 6e6, 6.9e6, 7.5e6, 9.3e6]
# create model dir
model_dir3 = [] # trained with 3 users
for number in from3_number:
    path_str = from3_prefix + str(int(number/100)) + ".ckpt"
    model_dir3.append(path_str)
model_dir4 = [] # trained with 4 users
for number in from4_number:
    path_str = from4_prefix + str(int(number/100)) + ".ckpt"
    model_dir4.append(path_str)
model_dir5 = [] # trained with 5 users
for number in from5_number:
    path_str = from5_prefix + str(int(number/100)) + ".ckpt"
    model_dir5.append(path_str)

model_dir = []
model_dir.append(model_dir3)
model_dir.append(model_dir4)
model_dir.append(model_dir5)

log_prefix = "/home/t-chengluo/xuyi-MAPPO/LC-MAPPO/results/abr_sat/time/ppo/2000mstest_hz128_usesaved/console_log/"
log_middle = ["run4", "run5", "run6"]
for indx_user in range(3):
    usernum = users[indx_user]
    for indx_model in range(3):
        model_list = model_dir[indx_model]
        for model in model_list:
            log_path = log_prefix + str(usernum) + "_" + log_middle[indx_model] + "/" + model[-10:-5] + ".txt"
            os.system('./test_time.sh '+str(usernum)+' '+model+" "+log_path)