from sklearn.feature_extraction.text import HashingVectorizer
from nltk.corpus import stopwords
import pandas as pd

def load_data(path = '../Data/issues.csv'):
    """
    path 对应文件中最后一列为类别 -> 前面的均为特征
    :param path: 数据文件路径
    :return: feature 和 label
    """
    with open(path,'r') as f:
        #初始化,x、y分别为 instance 和 label
        x,y = [],[]
        #读取数据进 x、y
        data = pd.read_csv(f,header=None)
        cols = data.columns.values.tolist()
        x = data[0]; y = data[1]
        #读取停用词，并过滤，最后还原成字符串
        word_filter = stopwords.words('english')
        for i in range(len(x)):
            z = x[i].split()
            z = [word for word in z if word not in word_filter]
            x[i] = " ".join(z)
        hv = HashingVectorizer(n_features=1000)
        x = hv.fit_transform(x).toarray()
    return x, y

