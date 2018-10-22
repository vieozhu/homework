# coding:utf-8
#
# 调用sklearn库实现svm算法
#
# reference from: https://www.cnblogs.com/luyaoblog/p/6775342.html
# ==================================================================


from sklearn import svm
from sklearn import model_selection
import numpy as np

root_path = './data/heart.txt'
test_path = './data/test.txt'


def file_read(data_path):
    """文件读入
    :param data_path: 数据集文件
    :return:返回列表
    """
    data = np.loadtxt(data_path, dtype=float, delimiter=' ')
    # print(data)
    return data


def data_split(data):
    """
    :param data:读入列表
    :return:数据集划分为训练集和验证集,x是特征y是类别标志
    """
    x, y = np.split(data, (13,), axis=1)
    # x = x[:, :2]  # 方便画图，只取前两列特征值向量训练
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x, y, random_state=1, train_size=0.7)

    # print x_test
    return x_train, x_test, y_train, y_test


def svm_train(test_data, x_train, x_test, y_train, y_test):
    """支持向量机
    svc二分类
    scr曲线你和函数回归

    kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）
    kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。

    decision_function_shape='ovr'时，为one v rest，即一个类别与其他类别进行划分，
    decision_function_shape='ovo'时，为one v one，即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。
    """

    clf = svm.SVC(C=0.08, kernel="linear", decision_function_shape="ovr")  # 线性核分类器
    # clf = svm.SVC(C=0.8, kernel="rbf", gamma=20, decision_function_shape="ovr")  # 高斯核分类器

    clf.fit(x_train, y_train.ravel())  # training the svc model

    # 计算准确率
    print("训练集准确率:")
    print(clf.score(x_train, y_train))  # 训练集准确率，linear核参数0.08达到0.86772
    print("测试集准确率:")
    print(clf.score(x_test, y_test))  # 测试集准确率，linear核参数0.08达到0.85185

    # 外部测试
    print("predict:")
    print(clf.predict(test_data))


if __name__ == "__main__":
    print("start...")
    # 读入数据
    data = file_read(root_path)
    test_data = file_read(test_path)
    # 数据集划分
    x_train, x_test, y_train, y_test = data_split(data)
    # svm分类器
    svm_train(test_data, x_train, x_test, y_train, y_test)
