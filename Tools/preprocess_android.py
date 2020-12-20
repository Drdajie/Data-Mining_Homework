import csv
import pandas as pd

def load_data(path = '../Data/Android Malware Detection Data/Android Malware Detection Data.csv'):
    """
    path 对应文件中最后一列为类别 -> 前面的均为特征
    :param path: 数据文件路径
    :return: feature 和 label
    """
    with open(path,'r') as f:
        x,y = [],[]
        data = pd.read_csv(f,header=None)
        cols = data.columns.values.tolist()
        x = data[cols[:-1]]; y = data[cols[-1]]
        """data = csv.reader(f)
        for row in data:
            x.append(row[0:-1])
            y.append(row[-1])"""
    return x, y
