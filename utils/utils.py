import random
import torch
import torch.nn as nn
import numpy as np
import os
import time
from torch.autograd import Variable

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def to_var(var, device=0):
    if torch.is_tensor(var):
        # var = Variable(var)
        if torch.cuda.is_available():
            var = var.to(device)
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x, device), var)
        return var

def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def binary2graycode(binary):
    gray = []
    gray.append(binary[0])
    for i in range(1, len(binary)):
        if binary[i - 1] == binary[i]:
            g = 0
        else:
            g = 1
        gray.append(g)
    return gray

def convert_lon2binary(lon, size, round_num):
    # 负数转换二进制后带负号，此处取绝对值，避免计算错误
    lon = abs(lon)
    lon = round(lon, round_num) * (10**round_num)
    bin_lon = '{0:b}'.format(int(lon)).zfill(size) # lon_size
    bin_lon_list = [int(i) for i in bin_lon]

    # if self.configs.use_graycode:
    bin_lon_list = binary2graycode(bin_lon_list)

    assert len(bin_lon_list) == size, "ERROR"
    # return np.array(bin_lon_list,dtype=np.int8)
    return np.asarray(bin_lon_list)

def convert_lat2binary(lat, size, round_num):
    # 负数转换二进制后带负号，此处取绝对值，避免计算错误
    lat = abs(lat)
    lat = round(lat, round_num) * (10**round_num)
    bin_lat = '{0:b}'.format(int(lat)).zfill(size) # lat_size
    bin_lat_list = [int(i) for i in bin_lat]

    # if self.configs.use_graycode:
    bin_lat_list = binary2graycode(bin_lat_list)

    assert len(bin_lat_list) == size, "ERROR"
    # return np.array(bin_lat_list,dtype=np.int8)
    return np.asarray(bin_lat_list)

def timestamp2timetuple(timestamp):
    timestamp = time.localtime(timestamp)

    return [timestamp.tm_wday,timestamp.tm_yday, timestamp.tm_hour * 60 + timestamp.tm_min]


def classify_time(time_info):
    # input: [timestamp.tm_wday,timestamp.tm_yday, timestamp.tm_hour * 60 + timestamp.tm_min]
    tm_wday, _, total_minutes = time_info

    # 判断是否是工作日
    is_weekday = 0 <= tm_wday <= 4

    # 判断是否是 Morning peak
    if is_weekday and 420 <= total_minutes < 540:
        return 0  # Morning peak

    # 判断是否是 Afternoon peak
    if is_weekday and 960 <= total_minutes < 1140:
        return 1  # Afternoon peak

    # 其他时间都是 Off-peak
    return 2  # Off-peak

def truncated_rand(mu = 0, sigma = 0.1, factor = 0.001, bound_lo = -0.01, bound_hi = 0.01):
    # using the defaults parameters, the success rate of one-pass random number generation is ~96%
    # gauss visualization: https://www.desmos.com/calculator/jxzs8fz9qr?lang=zh-CN
    while True:
        n = random.gauss(mu, sigma) * factor
        if bound_lo <= n <= bound_hi:
            break
    return n

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # 递归展平子列表
        else:
            flat_list.append(item)  # 直接添加非列表元素
    return flat_list

def GPSShift(src):
    #guassian (0,0.1)=> [-0.43701718299113373, 0.3855528305711887]
    return np.asarray([[p[0] + truncated_rand(mu=0, sigma=0.1,factor=0.001,bound_lo=-0.01, bound_hi=0.01),
             p[1] + truncated_rand(mu=0, sigma=0.1,factor=0.001,bound_lo=-0.01, bound_hi=0.01)] for p in src])

def SEGRandomMask(src: np.array, seg_len: int, seg_pad_ratio: float, pad_value: int):
    src = src.copy()
    replaces = np.where(np.random.random(seg_len) <= seg_pad_ratio)
    src[replaces] = np.asarray([pad_value] * len(replaces))   # 此处直接赋值会改变dataset原始值
    return src

def TimeSlide(timestamp: int):
    # half possibility  [+-30min], half possibility [+-2week]
    if random.random() <= 0.5:
        timestamp += random.randint(-1800,1800)
    else:
        timestamp += random.choice([1209600, 604800,0,-604800, -1209600])
    return timestamp2timetuple(timestamp)