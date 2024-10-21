import numpy as np
import geopandas as gpd
from pycitysim.map import Map
from shapely.geometry import box, Polygon, Point

from utils import *
from constant import RECTANGLE


def getGridId(longlat, lng_min, lat_min, lng_int, lat_int, gridNum):
    long = longlat[0]
    lat = longlat[1]
    i = (long - lng_min) // lng_int
    j = (lat - lat_min) // lat_int
    id = int(i*gridNum+j)
    return id

def getOD(longlats, rectangle, gridNum):
    length = gridNum ** 2
    lng_min, lat_min, lng_max, lat_max = rectangle
    lat_int = (lat_max - lat_min) / gridNum
    lng_int = (lng_max - lng_min) / gridNum

    od = np.zeros((length, length))
    for trace in longlats:
        for i in range(1, len(trace)):
            id0 = getGridId(trace[i-1], lng_min, lat_min, lng_int, lat_int, gridNum)
            id1 = getGridId(trace[i], lng_min, lat_min, lng_int, lat_int, gridNum)
            try:
                od[id0][id1] += 1
            except:
                pass

    sums = np.sum(od, axis=0, keepdims=True)
    sums[sums == 0] = 1
    normalized_arr = od / sums                  
    return normalized_arr


def getGridID_polygon(lnglat, polygon):
    pass

def getOD_polygon(longlats, polygons):
    length = len(polygons)
    od = np.zeros((length, length))
    for trace in tqdm(longlats):
        for i in range(1, len(trace)):
            
            id0 = find_grid_id(polygons, trace[i-1])
            id1 = find_grid_id(polygons, trace[i])
            if id0 is not None and id1 is not None:
                od[id0][id1] += 1
                # print(id0, id1)

    sums = np.sum(od, axis=0, keepdims=True)
    sums[sums == 0] = 1
    normalized_arr = od / sums                  
    return normalized_arr, od


def eval_population(city, folder, map):
    # load real data
    if city == "Beijing":

        with open("polygons/Beijing_user_sampled_1000_polygons.pkl", "rb") as f:
            polygons = pickle.load(f)
        with open("ground_truth/beijing/Beijing_user_sampled_1000_OD_norm.pkl", "rb") as f:
            realOD_norm = pickle.load(f)
        profile_path = None

    elif city in ['Newyork', 'San_Francisco']:  # Safegraph
        # 读取几何数据
        with open(f"polygons/{city}_polygons.pkl", "rb") as f:
            polygons = pickle.load(f)
        with open(f"polygons/{city}_cbg2index.pkl", "rb") as f:
            cbg_to_index = pickle.load(f)
        length = len(polygons)

        profile_path=f"profiles/{city}_user_sampled_1000.pkl"

        # 读取移动数据
        with open(f"ground_truth/safegraph/{city}_flow_2023-06-all.pkl", "rb") as f:
            cbg_flow = pickle.load(f)   # dict[(cbg_out, cbg_in): visitor]
        print("read real data ok", len(cbg_flow))

        # 计算 OD 矩阵
        od = np.zeros((length, length))
        for k, v in cbg_flow.items():
            cbg_out, cbg_in = k
            visitor_count = v
            try:
                id0 = cbg_to_index[cbg_out]
                id1 = cbg_to_index[cbg_in]
                od[id0][id1] += visitor_count
            except:
                pass
        sums = np.sum(od, axis=0, keepdims=True)
        sums[sums == 0] = 1
        normalized_arr = od / sums                  
        realOD_norm = normalized_arr
        print("gen real OD ok")

    elif city in ['London', 'Tokyo', 'Paris', 'Sydney', 'Sao_Paulo', 'Nairobi']:    # Foursquare
        # 读取几何数据
        # with open(f"res_new/{city}_polygons.pkl", "rb") as f:
        #     polygons = pickle.load(f)
        rec_dict = {
            "Sydney": [150.839882, -34.0432, 151.243988, -33.721288],
            "Paris": [2.24924, 48.812177, 2.423433, 48.903659],
            "London": [-0.315578, 51.3598, 0.107305, 51.614099]
        }
        rec = rec_dict[city]
        polygons = create_grid(rec[0], rec[1], rec[2], rec[3], 25)

        if city == "Sydney":
            profile_path=f"profiles/{city}_user_sampled_537.pkl"
            sampled = "_sampled_537_after_fill"
        else:
            profile_path=f"profiles/{city}_user_sampled_1000.pkl"
            sampled="_sampled_1000_after_fill"

        # 读取移动数据，计算OD
        if not os.path.exists(f"temp_res/res/{city}_OD{sampled}.pkl"):
            tracesR = readFoursquare(city, map,sampled=sampled, profile_path=profile_path)
            realData = genDataProcessForOD(tracesR)
            realOD_norm, realOD = getOD_polygon(realData, polygons)
        else:
            with open(f"temp_res/res/{city}_OD.pkl", "rb") as f:
                realOD = pickle.load(f)
    else:
        print(f"The City {city} is not support to eval individual mobility data. Only the following cities: 'Beijing', 'London', 'Tokyo', 'Paris', 'Sydney', 'Sao_Paulo', 'Nairobi', 'Newyork', 'San_Francisco'.")
        return 0

    print("Real data OD loaded!")

    # load generated data
    traces = readGenTraces(folder, map, profile_path=profile_path)
    genData = genDataProcessForOD(traces)
    genOD_norm, genOD = getOD_polygon(genData, polygons)

    with open(f"eval_res/{city}_genOD.pkl", "wb") as f:
        pickle.dump(genOD, f)

    # print(genOD)
    print("Generated data OD loaded!")

    # calculate OD similarity MSE
    loss = np.mean((realOD_norm - genOD_norm) ** 2)
    print("OD similarity (MSE)")
    print(loss)

    return loss, genOD


if __name__ == "__main__":
    import argparse
    from constant import MAP_PATH

    # 创建解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='Beijing', help='City name')
    parser.add_argument('--results', type=str, default='./output', help='Results directory')


    # 解析参数
    args = parser.parse_args()
    city = args.city
    folder = args.results

    if city not in MAP_PATH.keys():
        print(f"City '{city}' have no map supported, we support these map: {list(MAP_PATH.keys())}")
        exit(0)
    else:
        map_path = MAP_PATH[city]
    

    print("Start loading map ...")
    map = Map(
            mongo_uri="mongodb://sim:FiblabSim1001@mgo.db.fiblab.tech:8635/",
            mongo_db= map_path.split('.')[0],
            mongo_coll= map_path.split('.')[1],
            cache_dir="./cache/",
        )
    print('Map loaded!')

    eval_population(city, folder, map)