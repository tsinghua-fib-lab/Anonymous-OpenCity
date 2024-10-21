
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pycitysim.map import Map 
from datetime import datetime
from calSegForEval import calSegForEval
from shapely import Point


CONSUMPTION_INDEX = {
    "low": 1,
    "slightly_low": 2,
    "median": 3,
    "slightly_high": 4,
    "high": 5,
    "uncertain": 'n'
}

def timeInday(time):
    try:
        h,m = time.split(':')
    except:
        h,m,_ = time.split(':')
    h = int(h)
    m = int(m)   
    minutes = h*60+m
    return minutes/(24*60)

def timeSplit(time):
    time = time[1:-1]
    start, end = time.split(',')
    start = start.strip()
    end = end.strip()
    return (timeInday(start), timeInday(end))

def genDataProcess(trace, profile, map:Map, user_id=None):
    res = []
    for item in trace:
        poiid = item[2][1]
        poi = map.get_poi(poiid)
        # if poi is None:
        #     print(poiid)
        # print(poi)
        xy = poi['position']
        position_xy = (xy['x'], xy['y'])
        position_lnglat = map.xy2lnglat(xy['x'], xy['y'])
        SEtime = timeSplit(item[1])
        if type(profile) == list:
            if type(profile[2])==str:
                consumption = profile[2]
                try:
                    consumption = CONSUMPTION_INDEX[consumption]
                except:
                    print(consumption)
                    consumption = CONSUMPTION_INDEX[consumption.strip().replace(' ', '_')]
            else:
                consumption = float(profile[2])
        else:
            consumption = profile['consumption']
            if type(consumption) is str:
                try:
                    consumption = CONSUMPTION_INDEX[consumption]
                except:
                    consumption = CONSUMPTION_INDEX[consumption.replace(' ', '_')]
        res.append([SEtime, poiid, position_xy, position_lnglat, consumption, user_id])
    return res

def readGenTraces(file_path:str, map:Map, profile_path=None):  

    traces = []

    all_files = os.listdir(file_path)

    traj_files = [f for f in all_files if "traj" in f]

    if profile_path is not None:
        with open(profile_path, "rb") as f:
            profiles = pickle.load(f)
    
    print(len(traj_files))
    
    success = 0
    for file in tqdm(traj_files):
        # try:
        if 1:
            with open(os.path.join(file_path, file), 'r', encoding='UTF-8') as f:
                oneTrace = json.load(f)
            user_id = file.strip().split('_')[0]
            if profile_path is None:
                with open(os.path.join(file_path, file.replace('traj', 'profile')), 'r', encoding='UTF-8') as f:
                    profile = json.load(f)
            else:
                try:
                    profile = profiles[user_id]['profile']
                except:
                    profile = profiles[int(user_id)]['profile']
            trace = genDataProcess(oneTrace, profile, map, user_id)
            traces.append(trace)
            success += 1
        # except Exception as err:
        #     print(file, err)
        #     pass
        
    print("read all num: {}".format(success))
    print("actually all num: {}".format(len(traj_files)))
    # [(st, et), poiid, xy, lnglat, profile]
    return traces



def fqDataProcess(trace, poiDf:pd.DataFrame, map:Map, profile=0, user=None):
    # [SEtime, poiid, position(x, y), position(lng,lat)]
    res = []
    for i, item in enumerate(trace):
        poiid = item[1]
        poi = poiDf[poiDf['Venue ID'] == poiid].reset_index()
        # print(poi)
        position = (poi["Longitude"].iloc[0], poi["Latitude"].iloc[0])
        if map: position_xy = map.lnglat2xy(position[0], position[1])
        else: position_xy = (0,0)
        start_time, _ = trace[i]
        start_time = datetime.strptime(start_time, "%a %b %d %H:%M:%S %z %Y")
        if i+1 < len(trace):
            end_time, _ = trace[i + 1]
            end_time = datetime.strptime(end_time, "%a %b %d %H:%M:%S %z %Y")
            # 格式化时间为(10:20, 12:45)的形式
            time_range = (start_time.strftime("%H:%M"), end_time.strftime("%H:%M"))
        else:
            time_range = (start_time.strftime("%H:%M"), "23:59")

        SEtime = timeSplit(str(time_range).replace('\'',''))
        if type(profile) == list:
            if type(profile[2])==str:
                consumption = CONSUMPTION_INDEX[profile[2]]
            else:
                consumption = float(profile[2])
        else:
            consumption = profile['consumption']
            if type(consumption) is str:
                consumption = CONSUMPTION_INDEX[consumption]
        
        res.append([SEtime, poiid, position_xy, position,consumption,  user])
    return res


def readFoursquare(city, map, sampled='', rewrite=False, profile_path=None):
    if not rewrite and os.path.exists(f"cache/foursquare_{city}_real_data{sampled}.pkl"):
        with open(f"cache/foursquare_{city}_real_data{sampled}.pkl", "rb") as f:
            traces = pickle.load(f)
        return traces

    traj_folder = f"ground_truth/foursquare/{city}_checkins_traj{sampled}.json"
    with open(traj_folder, "r") as f:
        dat = json.load(f)

    poi_folder = f"ground_truth/foursquare/{city}_filtered_pois.csv"
    poiInfo = pd.read_csv(poi_folder)

    if profile_path is not None:
        with open(profile_path, "rb") as f:
            profiles = pickle.load(f)

    traces = []
    for user, date in dat.items():
        if profile_path is not None:
            profile = profiles[user]['profile']
        else:
            profile = 0.0
        # 将字符串时间转换为datetime对象
        for date_str, timestamps in date.items():
            date[date_str].sort(key=lambda x: datetime.strptime(x[0], "%a %b %d %H:%M:%S %z %Y"))

            trace = fqDataProcess(timestamps, poiInfo, map, profile, user)
            traces.append(trace)

    print(f"{len(traces)} foursquare trajectories Loaded.") 
    with open(f"cache/foursquare_{city}_real_data{sampled}.pkl", "wb") as f:
        pickle.dump(traces, f)  
    return traces


def genDataProcessForOD(traces):
    """
    将data 转换为 OD计算需要的格式，
    输入是 readgendata()的输出，或readFoursquare() 输出 traces
    """
    res = []
    for trace in traces:
        # res.append([SEtime, poiid, position_xy, position_lnglat])
        tempRes = []
        for i, t in enumerate(trace):
            tempRes.append(np.array(t[3]))
        # print(len(tempRes))
        res.append(np.array(tempRes))
    return res


def genDataProcessForCbgSeg(traces, polygons):
    """
    将gendata 处理为计算CBG segregation 所需格式
    """
    genDataForSeg = dict()
    id_count = {}
    for row in traces:
        for t in row:
            lng, lat = t[3]
            ids = find_grid_id(polygons, (lng, lat))
            if ids is None:
                continue
            profile = t[4]
            if ids not in id_count.keys():
                id_count[ids] = 1
            else:
                id_count[ids] += 1
            if ids not in genDataForSeg.keys():
                genDataForSeg[ids] = {}
            if profile not in genDataForSeg[ids].keys():
                genDataForSeg[ids][profile] = 1
            else:
                genDataForSeg[ids][profile] += 1
    
    filter_number = 0
    filtered_cbg = []
    for i, count in id_count.items():
        if count >= filter_number:
            filtered_cbg.append(i)

    # print(f"有{len(filtered_cbg)} 个CBG有不少于{4}人访问。")

    # 将genDataForSeg 转换为pd.Dataframe, columns = ['poiid', 'profiles']
    data = []
    sp_count = 0
    for poiid, profile in genDataForSeg.items():
        if poiid not in filtered_cbg: continue
        if len(profile) > 1 : pass # print("Here", profile)
        else: sp_count += 1
        for income, count in profile.items():
            data.append({'id':poiid, 'income': income, 'count': count})
    
    genDataForSegDf = pd.DataFrame(data)

    # print(f"共有{len(genDataForSeg)} CBGs 被访问，有{len(genDataForSegDf['id'].unique())} 个CBGs 超过{filter_number} 人访问过，有{sp_count}个CBGs 仅被一种收入（消费）水平的人访问过")

    genDataForSegDf.to_csv("eval_res/genData_forSeg.csv")

    genDataIncomeSegDf = calSegForEval(genDataForSegDf)

    return genDataForSegDf, genDataIncomeSegDf





def genDataProcessForSeg(traces):
    """
    将gendata ([SEtime, poiid, position_xy, position_lnglat, consumption]) 处理为计算seg 所需格式
    """

    genDataForSeg = dict()
    #   # [(st, et), poiid, xy, lnglat, profile]

    for row in traces:
        for t in row:
            # print(t)
            poiid = t[1]
            profile = t[4]
            if poiid not in genDataForSeg.keys():
                genDataForSeg[poiid] = {}
            if profile not in genDataForSeg[poiid].keys():
                genDataForSeg[poiid][profile] = 1
            else:
                genDataForSeg[poiid][profile] += 1

    # 将genDataForSeg 转换为pd.Dataframe, columns = ['poiid', 'profiles']
    data = []
    sp_count = 0
    for poiid, profile in genDataForSeg.items():
        if len(profile) > 1 : pass # print("Here", profile)
        else: sp_count += 1
        for income, count in profile.items():
            data.append({'id':poiid, 'income': income, 'count': count})
    
    genDataForSegDf = pd.DataFrame(data)

    print(f"共有{len(genDataForSeg)} poi 被访问，有{sp_count}个poi 仅被一种收入（消费）水平的人访问过")

    genDataForSegDf.to_csv("eval_res/genData_forSeg.csv")

    genDataIncomeSegDf = calSegForEval(genDataForSegDf)

    return genDataForSegDf, genDataIncomeSegDf

def cal_mse(a, b):
    """
    计算均方误差 (MSE)。

    参数:
    a -- numpy数组，真实值
    b -- numpy数组，预测值

    返回:
    mse -- 计算得到的均方误差
    """
    mse = np.mean((a - b) ** 2)
    return mse

def cal_mae(a, b):
    """
    计算平均绝对误差 (MAE)。

    参数:
    a -- numpy数组，真实值
    b -- numpy数组，预测值

    返回:
    mae -- 计算得到的平均绝对误差
    """
    mae = np.mean(np.abs(a - b))
    return mae


from shapely.geometry import box

def create_grid(xmin, ymin, xmax, ymax, grid_size):
    """
    创建网格并返回每个网格的多边形对象和ID。

    参数:
    xmin, ymin, xmax, ymax -- 矩形框的四个坐标
    grid_size -- 行（列）网格数量

    返回:
    grid_polygons -- 包含网格多边形和ID的列表
    """
    # 计算网格的数量
    y_int = (ymax - ymin) / grid_size
    x_int = (xmax - xmin) / grid_size

    num_x = grid_size
    num_y = grid_size

    # 创建网格
    grid_polygons = []
    for i in range(num_x):
        for j in range(num_y):
            poly_xmin = xmin + i * x_int
            poly_xmax = min(xmax, poly_xmin + x_int)
            poly_ymin = ymin + j * y_int
            poly_ymax = min(ymax, poly_ymin + y_int)

            polygon = box(poly_xmin, poly_ymin, poly_xmax, poly_ymax)
            grid_polygons.append((i * num_y + j, polygon))

    return grid_polygons


def find_grid_id(grid_polygons, p):
    """
    判断给定点在哪个网格内，并返回网格的ID。

    参数:
    grid_polygons -- 包含网格多边形和ID的列表
    x_0, y_0 -- 给定点的坐标

    返回:
    grid_id -- 网格的ID，如果点不在任何网格内，则返回None
    """
    point = Point(p[0], p[1])
    for grid_id, polygon in grid_polygons:
        # print(grid_id, polygon)
        if polygon.contains(point):
            return grid_id
    return None


def cal_gyration_radius(traces):
    """
    计算回转半径
    """

    points = []
    for row in traces:
        for t in row:
            points.append(t[2])
    centroid = np.mean(points, axis=0)

    # 计算每个点到质心的距离
    distances = np.sqrt(((points - centroid) ** 2).sum(axis=1))

    # 计算回转半径，这里我们使用所有距离的平均值
    gyration_radius = np.mean(distances)

    return gyration_radius