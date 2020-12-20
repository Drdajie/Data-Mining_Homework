from sklearn import svm
import time
import Tools.preprocess_issues as pre
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    time_start = time.time()
    x,y = pre.load_data()
    train_x,test_x,train_y,test_y = train_test_split(x,y)
    clf = svm.SVC()
    clf.fit(train_x,train_y)
    result = clf.score(test_x,test_y)
    time_end = time.time()
    print('issues数据集上，SVM正确率为：', result)
    print('issues数据集上，SVM运行结果为：', time_end - time_start)