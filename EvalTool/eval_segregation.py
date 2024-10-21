from pycitysim.map import Map 
from utils import *
import scipy


def segregation_evaluation(gendata, realdata):
    """
    计算 segregation index 的MSE
    """

    # 找出相同CBG
    # print(f"生成数据seg 均值：{np.mean(gendata['seg5'].to_numpy())}, 真实数据seg 均值: {np.mean(realdata['seg5'].to_numpy())}")
    common_rows = pd.merge(gendata, realdata, left_on='id', right_on='id')
    # print(f"{len(common_rows)} 个共同CBG")
    # print(common_rows)

    seg4_1 = common_rows['seg4_x'].to_numpy()
    seg4_2 = common_rows['seg4_y'].to_numpy()
    seg4_mse = cal_mse(seg4_1, seg4_2)

    seg5_1 = common_rows['seg5_x'].to_numpy()
    seg5_2 = common_rows['seg5_y'].to_numpy()
    seg5_mse = cal_mse(seg5_1, seg5_2)

    seg10_1 = common_rows['seg10_x'].to_numpy()
    seg10_2 = common_rows['seg10_y'].to_numpy()
    seg10_mse = cal_mse(seg10_1, seg10_2)

    print("seg4_mse/ seg5_mse/ seg10_mse")
    print("{:.4f} / {:.4f} / {:.4f}".format(seg4_mse, seg5_mse, seg10_mse))

    return seg4_mse, seg5_mse, seg10_mse


def eval_segregation(city:str, folder:str, map:Map):

    if city == "Beijing":
        with open("polygons/Beijing_user_sampled_1000_polygons.pkl", "rb") as f:
            polygons = pickle.load(f)
        realdataDf = pd.read_csv("ground_truth/beijing/Beijing_user_sampled_1000_Seg.csv")
        profile_path = None

    elif city in ['Newyork', 'San_Francisco']:
        # 加载几何数据
        with open(f"polygons/{city}_polygons.pkl", "rb") as f:
            polygons = pickle.load(f)
        with open(f"polygons/{city}_cbg2index.pkl", "rb") as f:
            cbg_to_index = pickle.load(f)
        length = len(polygons)
        # profile_path=f"profiles/{city}_user_sampled_1000.pkl"
        profile_path = None

        # 加载真实数据
        realFolder = f"ground_truth/safegraph/{city}_seg_cbg_z_n_0928.csv"
        realdataDf = pd.read_csv(realFolder)
        realdataDf['id'] = realdataDf['id'].apply(lambda x: cbg_to_index[x])
    elif city in ['London', 'Tokyo', 'Paris', 'Sydney', 'Sao_Paulo', 'Nairobi']:    # Foursquare
        # 读取几何数据
        # with open(f"res_new/{city}_polygons.pkl", "rb") as f:
        #     polygons = pickle.load(f)
        rec_dict = {
        "Sydney": [150.839882, -34.0432, 151.243988, -33.721288],
        "Paris": [2.24924, 48.812177, 2.423433, 48.903659],
        "London": [-0.315578, 51.3598, 0.107305, 51.614099]}
        rec = rec_dict[city]
        polygons = create_grid(rec[0], rec[1], rec[2], rec[3], 25)
        if city == "Sydney":
            profile_path=f"profiles/{city}_user_sampled_537.pkl"
            sampled = "_sampled_537_after_fill"
        else:
            profile_path=f"profiles/{city}_user_sampled_1000.pkl"
            sampled="_sampled_1000_after_fill"
        tracesR = readFoursquare(city, map, sampled=sampled, profile_path=profile_path, rewrite=True)
        _, realdataDf= genDataProcessForCbgSeg(tracesR, polygons)
    else:
        return 0

    traces = readGenTraces(folder, map, profile_path=profile_path)
    _, genDataIncomeSegDf = genDataProcessForCbgSeg(traces, polygons)
    # _, genDataIncomeSegDf = genDataProcessForSeg(traces)
    genDataIncomeSegDf.to_csv(f"eval_res/{city}_genData_CBG_Segregation.csv")

    segregation_evaluation(genDataIncomeSegDf, realdataDf)



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

    eval_segregation(city, folder, map)
