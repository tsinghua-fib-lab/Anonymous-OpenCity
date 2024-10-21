from eval_individual import eval_individual
from eval_population import eval_population
from eval_segregation import eval_segregation

from pycitysim.map import Map 

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


    eval_individual(city, folder, map)
    eval_population(city, folder, map)
    eval_segregation(city, folder, map)