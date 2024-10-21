import numpy as np
import pandas as pd
import tqdm
import warnings
import ast
warnings.filterwarnings('ignore')

# 定义加权直方图函数
def weighted_hist(x, bins, range_val):
    hist, _ = np.histogram(x.income_percentile, bins=bins, range=range_val, weights=x['count'], density=True)
    return np.around(hist / bins, decimals=3)

# 定义不同的加权直方图函数
def weighted_hist_4(x):
    return weighted_hist(x, bins=4, range_val=(0, 1))

def weighted_hist_5(x):
    return weighted_hist(x, bins=5, range_val=(0, 1))

def weighted_hist_10(x):
    return weighted_hist(x, bins=10, range_val=(0, 1))

# 定义收入分隔指数
def seg(x, fraction):
    return np.sum(np.abs(x - fraction))

def seg4(x):
    return seg(x, 0.25) * 2 / 3

def seg5(x):
    return seg(x, 0.2) / 1.6

def seg10(x):
    return seg(x, 0.1) / 1.8

INCOME_PERCENTILE = [0, 0.19, 0.39, 0.59, 0.79, 0.99]

def calSegForEval(genDataForSegDf):
    # 计算收入百分位数
    genDataForSegDf['income_percentile'] = genDataForSegDf['income'].apply(lambda x: INCOME_PERCENTILE[x])

    # print("计算收入百分位数\n",genDataForSegDf )

    # print(genDataForSegDf['income_percentile'])

    # 按POI分组计算加权直方图和收入segregation指数，保留经纬度
    grouped = genDataForSegDf.groupby('id')

    # 分别计算加权直方图和segregation指数
    weighted_hist_4_result = grouped.apply(weighted_hist_4)
    weighted_hist_5_result = grouped.apply(weighted_hist_5)
    weighted_hist_10_result = grouped.apply(weighted_hist_10)

    aa = pd.DataFrame(weighted_hist_4_result)
    bb = pd.DataFrame(weighted_hist_5_result)
    cc = pd.DataFrame(weighted_hist_10_result)

    aa = pd.merge(aa, bb, on ='id')
    aa = pd.merge(aa, cc, on ='id')
    aa.columns = ['distribution_4', 'distribution_5', 'distribution_10']
    aa = aa.reset_index()
    # print("按访问人数加权计算4，5，10分位数\n", aa)

    # 计算segregation指数
    aa['seg4'] = aa['distribution_4'].apply(seg4)
    aa['seg5'] = aa['distribution_5'].apply(seg5)
    aa['seg10'] = aa['distribution_10'].apply(seg10)

    
    # print("计算segregation\n", aa)

    # aa.to_csv("res/genData_income_segregation.csv")

    return aa
