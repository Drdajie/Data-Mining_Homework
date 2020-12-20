from sklearn import neighbors
import time
import Tools.preprocess_android as pre
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    time_start = time.time()
    x,y = pre.load_data()
    train_x,test_x,train_y,test_y = train_test_split(x,y)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(train_x,train_y)
    result = clf.score(test_x,test_y)
    time_end = time.time()
    print('KNN正确率为：',result)
    print('KNN运行时间为：',time_end-time_start)