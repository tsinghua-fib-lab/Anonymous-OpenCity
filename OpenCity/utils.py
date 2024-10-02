import argparse
import json
import math
import pickle
import random
# -*- coding: utf-8 -*-
import sys
from datetime import datetime, timedelta
from shapely.geometry import shape, Polygon
from operator import itemgetter
from random import choice
import pandas as pd
from scipy.interpolate import interp1d

import numpy as np
import openai

from utils import *

def cut_map_range(map, nw:list, se:list):
    """cut the map by input range (POI), range include two elements:
    1. North-West: [lng, lat]
    2. South-East: [lng, lat]
    """
    nw = map.lnglat2xy(lng=nw[0], lat=nw[1])
    se = map.lnglat2xy(lng=se[0], lat=se[1])
    x_set = [nw[0], se[0]]
    y_set = [se[1], nw[1]]
    pois = {}
    for poi in map.pois.values():
        if poi['position']['x'] > x_set[0] and poi['position']['x'] < x_set[1] and poi['position']['y'] > y_set[0] and poi['position']['y'] < y_set[1]:
            pois[poi['id']] = poi
    map.pois = pois
    (
        map._poi_tree,
        map._poi_list,
        map._driving_lane_tree,
        map._driving_lane_list,
        map._walking_lane_tree,
        map._walking_lane_list,
    ) = map._build_geo_index()

def get_polygon_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    polygons = []
    for feature in data['features']:
        geom = feature['geometry']
        polygon = shape(geom)  # 使用 shapely 的 shape() 函数将GeoJSON转换为Polygon
        polygons.append(polygon)
    return polygons[0]

def cut_map_poly(map_, container_poly):
    """
    Cut the map by provided polygon
    """
    pois = list(map_.pois.values())  # Point
    aois = list(map_.aois.values())  # Polygon
    contained_pois = {}
    contained_aois = {}
    for poi in pois:
        if container_poly.contains(poi['shapely_lnglat']):
            contained_pois[poi['id']] = poi
    for aoi in aois:
        if container_poly.contains(aoi['shapely_lnglat']):
            contained_aois[aoi['id']] = aoi
    map_.pois = contained_pois
    map_.aois = contained_aois
    (
        map_._poi_tree,
        map_._poi_list,
        map_._driving_lane_tree,
        map_._driving_lane_list,
        map_._walking_lane_tree,
        map_._walking_lane_list,
    ) = map_._build_geo_index()


def getEventNum():
    '''
    通过真实概率值采样,得到
    '''
    value = [3, 5, 4, 8, 6, 7, 10, 9, 14, 16, 13, 11, 12, 17, 15, 21]
    prob = [0.5162918266593527, 0.1417443773998903, 0.20065825562260012, 0.019199122325836534, 0.06297312122874382, 0.03708173340647285, 0.005156335710367526, 0.00811848601206802, 0.0009873834339001646, 0.0005485463521667581, 0.0012068019747668679, 0.0029621503017004938, 0.0020844761382336806, 0.00032912781130005483, 0.0005485463521667581, 0.00010970927043335162]
    index = list(range(len(value)))
    sample = np.random.choice(index, size=1, p=prob)  # 根据计算出来的概率值进行采样
    return value[sample[0]]

def getTimeDistribution():
    file = open('DataLocal/事件的时间分布/timeDistribution_Event.pkl','rb') 
    timeDistribution_Event = pickle.load(file)
    timeDistribution_Event['banking and financial services'] = timeDistribution_Event['handle the trivialities of life']
    timeDistribution_Event['cultural institutions and events'] = timeDistribution_Event['handle the trivialities of life']
    return timeDistribution_Event

def setIO(modeChoice, keyid):
    if modeChoice == 'labeld':
        file = open('DataLocal/人物profile统计无年龄/标注数据人物profile.pkl','rb') 
        profileDict = pickle.load(file)  # 标注数据是需要id的
        profileIds = list(profileDict.keys())
        profiles = list(profileDict.values())
        personNum = len(profiles)
        genIdList = [keyid]  # 对一类人生成5个模板,将key分配到id上进行加速
        loopStart = 0
        loopEnd = personNum
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    elif modeChoice == 'realRatio':
        file = open('DataLocal/人物profile统计无年龄/profileWordDict.pkl','rb') 
        profileDict = pickle.load(file)  # 其中的keys已经做了匿名化
        profileIds = list(profileDict.keys())
        profilesAndNum = list(profileDict.values())
        profiles = [item[0] for item in profilesAndNum]
        personNum = len(profiles)
        genIdList = [0] # list(range(10))  # 现在生成10次
        loopStart = keyid*60  # 将多key放到id上加速,top300其实也是可以的
        loopEnd = keyid*60+60
        return profileIds, profiles, personNum, genIdList, loopStart, loopEnd
    
    else:
        print('error!')
        sys.exit(0)


def getDay():
    '''
    随机采样今天是星期几
    '''
    value = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    prob = [0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1428, 0.1432]
    index = list(range(len(value)))
    sample = np.random.choice(index, size=1, p=prob)  # 根据计算出来的概率值进行采样
    return value[sample[0]]


def sampleTime(event):
    '''
    根据事件的类型,在真实数据统计出的分布中进行采样,获取时间
    '''
    # genEs = ["go to work", "go home", "eat", "do shopping", "do sports", "excursion", "leisure or entertainment", "go to sleep", "medical treatment", "handle the trivialities of life", "banking and financial services", "cultural institutions and events"]
    # realEs = ['excursion', 'leisure or entertainment', 'eat', 'go home', 'do sports', 'handle the trivialities of life', 'do shopping', 'go to work', 'medical treatment', 'go to sleep']
    # for item in genEs:
    #     if item not in realEs:
    #         print(item)

    # print(timeDistribution_Event.keys())
    timeDistribution_Event = getTimeDistribution()
    timeDis = timeDistribution_Event[event]
    timeZones = list(timeDis.keys())
    # print(timeZones)
    # sys.exit(0)
    length = len(list(timeDis.keys()))
    weightList = list(timeDis.values())
    indexx = list(range(length))
    sample = np.random.choice(indexx, size=1, p=weightList)  # 根据计算出来的概率值进行采样
    timeZone = timeZones[sample[0]]  # 选好了以半小时为度量的区间
    # print(timeZone)
    minutes = getTimeFromZone(timeZone)
    # print(minutes)
    return minutes


def add_time(start_time, minutes):
    """
    计算结束后的时间，给出起始时间和增量时间（分钟）
    :param start_time: 起始时间，格式为 '%H:%M'
    :param minutes: 增量时间（分钟）
    :return: 结束后的时间，格式为 '%H:%M'；是否跨越了一天的标志
    """
    # 将字符串转换为 datetime 对象，日期部分设为一个固定的日期
    start_datetime = datetime.strptime('2000-01-01 ' + start_time, '%Y-%m-%d %H:%M')

    # 增加指定的分钟数
    end_datetime = start_datetime + timedelta(minutes=minutes)

    # 判断是否跨越了一天
    if end_datetime.day != start_datetime.day:
        cross_day = True
    else:
        cross_day = False

    # 将结果格式化为字符串，只包含时间部分
    end_time = end_datetime.strftime('%H:%M')

    return end_time, cross_day

def getTimeFromZone(timeZone0):
    time0, time1 = timeZone0.split('-')
    time0 = float(time0)/2  # 这里已经化成小时了
    time1 = float(time1)/2
    # print(time0)
    # print(time1)
    
    sampleResult = random.uniform(time0, time1)  # 采样一个具体的时间值出来,单位是小时
    # print(sampleResult)  
    minutes = int(sampleResult*60)
    return minutes

def sampleGapTime():
    '''
    事件之间的间隔时间
    '''
    minutes = getTimeFromZone('0-2')  # 将事件的间隔时间设置为0到1个小时
    return minutes

def choiceProfile(city:str, select_cbgs:list[int] = None, number:int=1):
    if city == "beijing":
        profileIds, profiles, personNum, genIdList, loopStart, loopEnd = setIO(modeChoice='labeld',  keyid = 0)
        educations = []
        genders = []
        consumptions = []
        occupations = []
        for i in range(number):
            index = random.randint(0, 50)
            personBasicInfo = profiles[index]
            education, gender, consumption, occupation = personBasicInfo.split('-')
            educations.append(education)
            genders.append(gender)
            consumptions.append(consumption)
            occupations.append(occupation)
        return educations, genders, consumptions, occupations
    elif city == 'new_york':
        dex = [57443, 77615, 96307, 122718]
        consumption_level = ['low', 'slightly_low', 'median', 'slightly_high', 'high']
        df = pd.read_csv('./DataLocal/cbg_features_2019.csv')
        educations = []
        genders = []
        consumptions = []
        occupations = []
        for i in range(number):
            select = df[df['census_block_group']==select_cbgs[i]]
            # 平均收入和中位收入数据
            average_income = select['average_income'].values[0]
            median_income = select['median_income'].values[0]

            # 定义标准差
            if (average_income - median_income) > 0:
                std_dev = (average_income - median_income)
            else:
                std_dev = (median_income - average_income)

            # 生成一个从正态分布中采样的合理收入数值
            income_sample = np.random.normal(loc=average_income, scale=std_dev)
            consumption_index = -1
            for i in range(len(dex)):
                if income_sample < dex[i]:
                    consumption_index = i
            consumption = consumption_level[consumption_index]
            gender = 'female'
            if random.random() > select['female_ratio'].values[0]:
                gender = 'male'
            profileIds, profiles, personNum, genIdList, loopStart, loopEnd = setIO(modeChoice='labeld',  keyid = 0)
            index = random.randint(0, 50)
            personBasicInfo = profiles[index]
            education, _, _, occupation = personBasicInfo.split('-')
            educations.append(education)
            genders.append(gender)
            consumptions.append(consumption)
            occupations.append(occupation)
        return educations, genders, consumptions, occupations
    elif city == 'san_francisco':
        dex = [114215, 139413, 161969, 186726]
        consumption_level = ['low', 'slightly_low', 'median', 'slightly_high', 'high']
        df = pd.read_csv('./DataLocal/cbg_features_2019.csv')
        educations = []
        genders = []
        consumptions = []
        occupations = []
        for i in range(number):
            select = df[df['census_block_group']==select_cbgs[i]]
            # 平均收入和中位收入数据
            average_income = select['average_income'].values[0]
            median_income = select['median_income'].values[0]

            # 定义标准差
            if (average_income - median_income) > 0:
                std_dev = (average_income - median_income)
            else:
                std_dev = (median_income - average_income)

            # 生成一个从正态分布中采样的合理收入数值
            income_sample = np.random.normal(loc=average_income, scale=std_dev)
            consumption_index = -1
            for i in range(len(dex)):
                if income_sample < dex[i]:
                    consumption_index = i
            consumption = consumption_level[consumption_index]
            gender = 'female'
            if random.random() > select['female_ratio'].values[0]:
                gender = 'male'
            profileIds, profiles, personNum, genIdList, loopStart, loopEnd = setIO(modeChoice='labeld',  keyid = 0)
            index = random.randint(0, 50)
            personBasicInfo = profiles[index]
            education, _, _, occupation = personBasicInfo.split('-')
            educations.append(education)
            genders.append(gender)
            consumptions.append(consumption)
            occupations.append(occupation)
        return educations, genders, consumptions, occupations
    elif city == "paris":
        with open('DataLocal/city_profile.json', 'r') as f:
            profile_paris = json.load(f)
            profile_paris = profile_paris['Paris']
        # * income sample
        consumption_level = ['low', 'slightly_low', 'median', 'slightly_high', 'high']
        dex = [15455, 24345, 38647.5, 58362.5]
        percentiles = [10, 20, 40, 50, 60, 80, 90]
        income = [11010, 15455, 24345, 28790, 38647.5, 58362.5, 68220]
        # Generate uniform random numbers between 0 and 1
        uniform_randoms = np.random.rand(number) * 100  # Convert to percentage
        # Interpolate to generate samples based on the distribution
        sample_incomes = np.interp(uniform_randoms, percentiles, income)
        consumptions = []
        for income_sample in sample_incomes:
            consumption_index = -1
            for i in range(len(dex)):
                if income_sample < dex[i]:
                    consumption_index = i
            consumptions.append(consumption_level[consumption_index])

        # * gender sample
        male_rate = profile_paris['gender']['male']
        genders = ['male' if random.random() < male_rate else 'female' for _ in range(number)]

        # * occupation sample
        occupation_distribution = profile_paris['occupation']
        occupations_name = list(occupation_distribution.keys())
        probabilities = [value for value in occupation_distribution.values()]
        left_prob = 100 - sum(probabilities)
        occupations_name.append('Unknown')
        probabilities.append(left_prob)
        occupations = random.choices(occupations_name, probabilities, k=number)

        # * education sample
        education_distribution = profile_paris['education']
        education_level = list(education_distribution.keys())
        probabilities = list(education_distribution.values())
        left_prob = 1 - sum(probabilities)
        education_level.append('Unknown')
        probabilities.append(left_prob)
        educations = random.choices(education_level, probabilities, k=number)
        return educations, genders, consumptions, occupations
    elif city == "sydney":
        with open('DataLocal/city_profile.json', 'r') as f:
            profile_sydney = json.load(f)
            profile_sydney = profile_sydney['Sydney']
        # * consumption
        # Income thresholds based on the provided data
        low_income_threshold = 650
        high_income_threshold = 3000
        # Percentile breakdowns
        low_income_percentile = 0.142
        high_income_percentile = 1 - 0.358  # This means 64.2% have income <= 3000

        # Interpolation range between $650 and $3000 (assumed uniform distribution)
        middle_income_percentile_range = high_income_percentile - low_income_percentile

        # Function to calculate income percentiles
        def get_income_percentile(p):
            if p <= low_income_percentile:
                # Income below $650
                return low_income_threshold * p / low_income_percentile
            elif p <= high_income_percentile:
                # Income between $650 and $3000 (interpolation)
                return low_income_threshold + (p - low_income_percentile) * (high_income_threshold - low_income_threshold) / middle_income_percentile_range
            else:
                # Income above $3000 (linear growth assumed for simplicity)
                return high_income_threshold + (p - high_income_percentile) * 2000  # Scale for higher incomes
        # Generate percentiles
        percentiles = [0.2, 0.4, 0.6, 0.8]
        dex = [get_income_percentile(p) for p in percentiles]
        # Sampling incomes based on the same distribution
        def sample_income(n_samples):
            p_samples = np.random.rand(n_samples)
            income_samples = [get_income_percentile(p) for p in p_samples]
            return income_samples
        income_samples = sample_income(number)
        consumption_level = ['low', 'slightly_low', 'median', 'slightly_high', 'high']
        consumptions = []
        for income_sample in income_samples:
            consumption_index = -1
            for i in range(len(dex)):
                if income_sample < dex[i]:
                    consumption_index = i
            consumptions.append(consumption_level[consumption_index])
        
        # * gender
        male_rate = profile_sydney['gender']['male']
        genders = ['male' if random.random() < male_rate else 'female' for _ in range(number)]

        # * occupation
        occupation_distribution = profile_sydney['occupation']
        occupations_name = list(occupation_distribution.keys())
        probabilities = [value for value in occupation_distribution.values()]
        left_prob = 1 - sum(probabilities)
        if left_prob > 0:
            occupations_name.append('Unknown')
            probabilities.append(left_prob)
        occupations = random.choices(occupations_name, probabilities, k=number)

        # * educations
        education_distribution = profile_sydney['education']
        education_level = list(education_distribution.keys())
        probabilities = list(education_distribution.values())
        left_prob = 1 - sum(probabilities)
        if left_prob > 0:
            education_level.append('Unknown')
            probabilities.append(left_prob)
        educations = random.choices(education_level, probabilities, k=number)
        return educations, genders, consumptions, occupations
    elif city == "london":
        with open('DataLocal/city_profile.json', 'r') as f:
            profile_london = json.load(f)
            profile_london = profile_london['London']
        # * genders
        male_rate = profile_london['gender']['male']
        genders = ['male' if random.random() < male_rate else 'female' for _ in range(number)]

        # * occupations
        occupation_distribution = profile_london['occupation']
        occupations_name = list(occupation_distribution.keys())
        probabilities = [value for value in occupation_distribution.values()]
        left_prob = 1 - sum(probabilities)
        if left_prob > 0:
            occupations_name.append('Unknown')
            probabilities.append(left_prob)
        occupations = random.choices(occupations_name, probabilities, k=number)

        # * consumptions
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        incomes = [138, 259, 345, 439, 550, 667, 816, 1025, 1429]
        dex = [259, 439, 667, 1025]
        # Interpolate the percentiles to allow continuous sampling
        ecdf = interp1d(percentiles, incomes, kind='linear', fill_value="extrapolate")
        # Function to sample from the empirical distribution
        def sample_income_ecdf(n_samples):
            # Generate uniform random percentiles between 10th and 90th
            random_percentiles = np.random.uniform(10, 90, n_samples)
            # Map the random percentiles to income values using the ECDF
            sampled_incomes = ecdf(random_percentiles)
            return sampled_incomes
        # Generate 1000 income samples
        income_samples = sample_income_ecdf(number)
        consumption_level = ['low', 'slightly_low', 'median', 'slightly_high', 'high']
        consumptions = []
        for income_sample in income_samples:
            consumption_index = -1
            for i in range(len(dex)):
                if income_sample < dex[i]:
                    consumption_index = i
            consumptions.append(consumption_level[consumption_index])
        
        # * educations
        educations = []
        for i in range(number):
            if random.random() < profile_london['education'][genders[i]]['High education qualification']:
                educations.append('High education qualification')
            else:
                educations.append('Low education qualification')
        return educations, genders, consumptions, occupations
    else:
        raise Exception("Wrong City")

def choiceHW(city:str, map, number:int=1):
    if city == 'beijing':
        file = open('DataLocal/家和工作地汇总/hws.pkl','rb') 
        HWs = pickle.load(file)
        homes = []
        works = []
        for _ in range(number):
            while True:
                (home, work) = choice(HWs)
                if home[1] in map.pois.keys() and work[1] in map.pois.keys():
                    break
            homes.append(home)
            works.append(work)
        return (homes, works, None)
    elif city == 'new_york':
        homes = []
        works = []
        cbg_ids = []
        with open('DataLocal/newyork.geojson', 'r') as f:
            newyork = json.load(f)
        for i in range(number):
            # Home
            cbg_id = None
            while True:
                home = random.choice(list(map.pois.values()))
                for cbg in newyork['features']:
                    poly = Polygon(cbg['geometry']['coordinates'][0][0])
                    if poly.contains(home['shapely_lnglat']):
                        cbg_id = int(cbg['properties']['CensusBlockGroup'])
                        break
                if cbg_id != None:
                    break
            # Work
            work = random.choice(list(map.pois.values()))
            home = [home['name'], home['id']]
            work = [work['name'], work['id']]
            homes.append(home)
            works.append(work)
            cbg_ids.append(cbg_id)
        return (homes, works, cbg_ids)
    elif city == 'san_francisco':
        homes = []
        works = []
        cbg_ids = []
        with open('DataLocal/sf.geojson', 'r') as f:
            sf = json.load(f)
        for i in range(number):
            # Home
            cbg_id = None
            while True:
                home = random.choice(list(map.pois.values()))
                for cbg in sf['features']:
                    poly = Polygon(cbg['geometry']['coordinates'][0][0])
                    if poly.contains(home['shapely_lnglat']):
                        cbg_id = int(cbg['properties']['CensusBlockGroup'])
                        break
                if cbg_id != None:
                    break
            # Work
            work = random.choice(list(map.pois.values()))
            home = [home['name'], home['id']]
            work = [work['name'], work['id']]
            homes.append(home)
            works.append(work)
            cbg_ids.append(cbg_id)
        return (homes, works, cbg_ids)
    else:
        homes = []
        works = []
        for i in range(number):
            home = random.choice(list(map.pois.values()))
            work = random.choice(list(map.pois.values()))
            home = [home['name'], home['id']]
            work = [work['name'], work['id']]
            homes.append(home)
            works.append(work)
        return (homes, works, None)
    
def get_poi(sim, id):
    result = sim.map.get_poi(id)
    res = {}
    res['poi_id'] = result['id']
    res['aoi_id'] = result['aoi_id']
    res['name'] = result['name']
    res['x'] = result['position']['x']
    res['y'] = result['position']['y']
    (res['long'],  res['lati']) = sim.map.xy2lnglat(result['position']['x'], result['position']['y'])
    return res


def getDirectEventID(event):
    # 直接从event查询具体的POI, 也不再问具体的类目
    if event in ['have breakfast', 'have lunch', 'have dinner', 'eat']:
        return "10"
    elif event == 'do shopping':
        return "13"
    elif event == 'do sports':
        return "18"
    elif event == 'excursion':  # 这里指短期的旅游景点
        return "22"
    elif event == 'leisure or entertainment':
        return "16"
    elif event == 'medical treatment':
        return "20"
    elif event == 'handle the trivialities of life':
        return "14"
    elif event == 'banking and financial services':
        return "25"
    elif event == 'government and political services':
        return "12"
    elif event == 'cultural institutions and events':
        return "23"
    else:
        print('\nIn function event2cate: The selected choice is not in the range!\n')
        sys.exit(0)
        
def event2poi_gravity_matrix(sim, map, label, nowPlace):
    nowPlace = map.get_poi(nowPlace[1])  # 由poi id查询到poi的全部信息
    labelQueryId = label
    
    # 现在就假设是在10km内进行POI的选择,10km应该已经是正常人进行出行选择的距离上限了,再远就不对劲了。
    pois10k = sim.get_pois_from_matrix(
            center = (nowPlace['position']['x'], nowPlace['position']['y']), 
            prefix= labelQueryId, 
        )
    if len(pois10k) > 10000:
        pois10k = pois10k[:10000]
    
    if pois10k[-1][1] < 5000:
            pois10k = map.query_pois(
            center = (nowPlace['position']['x'], nowPlace['position']['y']), 
            radius = 10000,  # 10km查到5000个POI是一件非常轻松的事情
            category_prefix= labelQueryId, 
            limit = 30000  # 查询10000个POI
        )  # 得到的pois是全部的信息.
    
    N = len(pois10k)
    # 这么一番操作之后的POI数量应该会有很多了, 关于密度如何计算的问题
    # 关于POI的密度的问题，相对于我现在的位置而言，密度有多少。
    pois_Dis = {"1k":[], "2k":[], "3k":[], "4k":[], "5k":[], "6k":[], "7k":[], "8k":[], "9k":[], "10k":[], "more":[]}
    for poi in pois10k:
        iflt10k = True
        for d in range(1,11):
            if (d-1)*1000 <= poi[1] < d*1000:
                pois_Dis["{}k".format(d)].append(poi)
                iflt10k = False
                break
        if iflt10k:
            pois_Dis["more"].append(poi)
    
    res = []
    distanceProb = []
    for poi in pois10k:
        iflt10k = True
        for d in range(1,11):
            if (d-1)*1000 <= poi[1] < d*1000:
                n = len(pois_Dis["{}k".format(d)])
                S = math.pi*((d*1000)**2 - ((d-1)*1000)**2)
                density = n/S
                distance = poi[1]
                distance = distance if distance>1 else 1
                
                # 距离衰减系数,用距离的平方来进行计算,使得远方的地点更加不容易被选择 TODO 这里是平方倒数, 还是一次方倒数
                weight = density / (distance**2)  # 原来是(distance**2),权重貌似是合理的
                
                # 修改:需要对比较近的POI的概率值进行抑制
                # weight = weight * math.exp(-15000/(distance**1.6))  # 抑制近POI值
                
                res.append((poi[0]['name'], poi[0]['id'], weight, distance))
                
                # TODO 这里的一次选择的概率值也修改了,需要进一步check
                distanceProb.append(1/(math.sqrt(distance)))  # 原来是distanceProb.append(1/(math.sqrt(distance)))
                
                iflt10k = False
                break
    
    # 从中抽取50个.
    distanceProb = np.array(distanceProb)
    distanceProb = distanceProb/np.sum(distanceProb)
    distanceProb = list(distanceProb)
    
    options = list(range(len(res)))
    sample = list(np.random.choice(options, size=50, p=distanceProb))  # 根据计算出来的概率值进行采样
    
    get_elements = itemgetter(*sample)
    random_elements = get_elements(res)
    # printSeq(random_elements)
    # sys.exit(0)
    # random_elements = random.sample(res, k=30)
    
    # 接下来需要对权重归一化,成为真正的概率值
    weightSum = sum(item[2] for item in random_elements)
    final = [(item[0], item[1], item[2]/weightSum, item[3]) for item in random_elements]
    # printSeq(final)
    # sys.exit(0)
    return final
        
def event2poi_gravity(map, label, nowPlace):
    # 直接从意图对应到POI，好处是数量多
    # 这里还是考虑了POI类型的.
    
    nowPlace = map.get_poi(nowPlace[1])  # 由poi id查询到poi的全部信息
    labelQueryId = label
    
    # 现在就假设是在10km内进行POI的选择,10km应该已经是正常人进行出行选择的距离上限了,再远就不对劲了。
    pois10k = map.query_pois(
            center = (nowPlace['position']['x'], nowPlace['position']['y']), 
            radius = 10000,  # 10km查到5000个POI是一件非常轻松的事情
            category_prefix= labelQueryId, 
            limit = 20000  # 查询10000个POI
        )  # 得到的pois是全部的信息.
    
    if pois10k[-1][1] < 5000:
            pois10k = map.query_pois(
            center = (nowPlace['position']['x'], nowPlace['position']['y']), 
            radius = 10000,  # 10km查到5000个POI是一件非常轻松的事情
            category_prefix= labelQueryId, 
            limit = 30000  # 查询10000个POI
        )  # 得到的pois是全部的信息.
    
    N = len(pois10k)
    # 这么一番操作之后的POI数量应该会有很多了, 关于密度如何计算的问题
    # 关于POI的密度的问题，相对于我现在的位置而言，密度有多少。
    pois_Dis = {"1k":[], "2k":[], "3k":[], "4k":[], "5k":[], "6k":[], "7k":[], "8k":[], "9k":[], "10k":[], "more":[]}
    for poi in pois10k:
        iflt10k = True
        for d in range(1,11):
            if (d-1)*1000 <= poi[1] < d*1000:
                pois_Dis["{}k".format(d)].append(poi)
                iflt10k = False
                break
        if iflt10k:
            pois_Dis["more"].append(poi)
    
    res = []
    distanceProb = []
    for poi in pois10k:
        iflt10k = True
        for d in range(1,11):
            if (d-1)*1000 <= poi[1] < d*1000:
                n = len(pois_Dis["{}k".format(d)])
                S = math.pi*((d*1000)**2 - ((d-1)*1000)**2)
                density = n/S
                distance = poi[1]
                distance = distance if distance>1 else 1
                
                # 距离衰减系数,用距离的平方来进行计算,使得远方的地点更加不容易被选择 TODO 这里是平方倒数, 还是一次方倒数
                weight = density / (distance**2)  # 原来是(distance**2),权重貌似是合理的
                
                # 修改:需要对比较近的POI的概率值进行抑制
                # weight = weight * math.exp(-15000/(distance**1.6))  # 抑制近POI值
                
                res.append((poi[0]['name'], poi[0]['id'], weight, distance))
                
                # TODO 这里的一次选择的概率值也修改了,需要进一步check
                distanceProb.append(1/(math.sqrt(distance)))  # 原来是distanceProb.append(1/(math.sqrt(distance)))
                
                iflt10k = False
                break
    
    # 从中抽取50个.
    distanceProb = np.array(distanceProb)
    distanceProb = distanceProb/np.sum(distanceProb)
    distanceProb = list(distanceProb)
    
    options = list(range(len(res)))
    sample = list(np.random.choice(options, size=50, p=distanceProb))  # 根据计算出来的概率值进行采样
    
    get_elements = itemgetter(*sample)
    random_elements = get_elements(res)
    # printSeq(random_elements)
    # sys.exit(0)
    # random_elements = random.sample(res, k=30)
    
    # 接下来需要对权重归一化,成为真正的概率值
    weightSum = sum(item[2] for item in random_elements)
    final = [(item[0], item[1], item[2]/weightSum, item[3]) for item in random_elements]
    # printSeq(final)
    # sys.exit(0)
    return final



def sampleNoiseTime():
    noise = random.randint(-10, 10)
    return noise