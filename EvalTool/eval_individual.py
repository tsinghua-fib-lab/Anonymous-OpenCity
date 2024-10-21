from utils import *

from pycitysim.map import Map 

def cal_gyration_radius_new(traces):
    """
    计算回转半径
    """
    # [SEtime, poiid, position_xy, position_lnglat, comsumption, uid]
    radius_dict = {}
    for row in traces:
        points = []
        for t in row:
            user = t[5]
            points.append(t[2])
        if len(points):
            centroid = np.mean(points, axis=0)
            # 计算每个点到质心的距离
            distances = np.sqrt(((points - centroid) ** 2).sum(axis=1))
            # 计算回转半径，这里我们使用所有距离的平均值
            gyration_radius = np.mean(distances)
            radius_dict[user] = gyration_radius
    
    # 转换为dataframe 返回
    data = []
    for k, v in radius_dict.items():
        data.append({"user":str(k), "radius":v})
    radiusDf = pd.DataFrame(data)

    return radius_dict, radiusDf

def eval_radius(realDf, genDf):
    genDf['user'] = genDf['user'].astype(str)
    realDf['user'] = realDf['user'].astype(str)
    common_rows = pd.merge(genDf, realDf, left_on='user', right_on='user')
    r_1 = common_rows['radius_x'].to_numpy()/1000
    r_2 = common_rows['radius_y'].to_numpy()/1000
    # print(f"生成数据radius 均值：{np.mean(r_1)}, radius 均值: {np.mean(r_2)}")
    r_mse = cal_mse(r_1, r_2)
    return r_mse

def eval_individual(city:str, folder:str, map:Map):
    """
    eval individual moblity
    """

    # load groundtruth
    if city == "Beijing":
        with open("polygons/Beijing_user_sampled_1000_polygons.pkl", "rb") as f:
            polygons = pickle.load(f)
        realDf = pd.read_csv("ground_truth/beijing/Beijing_user_sampled_1000_radius.csv")
        profile_path = None
        # groundtruth = "ground_truth/beijing/realForEval_17825.pkl"
        # with open(groundtruth, "rb") as f:
        #     real_data = pickle.load(f)
        # print(f"{len(real_data)} tencent data loaded.")
    elif city in ['London', 'Tokyo', 'Paris']:
        profile_path=f"profiles/{city}_user_sampled_1000.pkl"
        traces = readFoursquare(city, map, rewrite=True,sampled="_sampled_1000_after_fill", profile_path=profile_path)
        _, realDf = cal_gyration_radius_new(traces)
    elif city in [ 'Sydney', 'Sao_Paulo', 'Nairobi']:
        profile_path=f"profiles/{city}_user_sampled_537.pkl"
        traces = readFoursquare(city, map, rewrite=True,sampled="_sampled_537_after_fill", profile_path=profile_path)
        _, realDf = cal_gyration_radius_new(traces)
    else:
        print(f"The City {city} is not support to eval individual mobility data. Only the following cities: 'Beijing', 'London', 'Tokyo', 'Paris', 'Sydney', 'Sao_Paulo', 'Nairobi'.")
        return 0
    
    # load gendata
    gen_data = readGenTraces(folder, map, profile_path=profile_path)
    print(f"{len(gen_data)} generated data loaded.")
    _,genDf = cal_gyration_radius_new(gen_data)
    genDf.to_csv(f"eval_res/{city}_genData_RadiusDf.csv")
    print("R_mse = ", eval_radius(realDf, genDf))
    

    return 
    

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

    print(city, folder)

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

    eval_individual(city, folder, map)