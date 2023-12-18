import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import model_selection

import joblib
import warnings
warnings.filterwarnings("ignore")
if __name__ == "__main__":
    # 读取特征向量数据集
    data = pd.read_csv(
        "D:\\desktop\\opencvcode\\teamquiz2.0\\assignment02\\data\\savedfile.csv", encoding="gbk")
    # 将类别转换为索引
    #data 是一个数据对象，可能是一个 DataFrame 或类似的数据结构。data["类别"] 表示选择名为 "类别" 的列，并返回一个包含该列所有元素的 Series 对象。
#然后，通过在 data["类别"] 后面再次使用方括号 [] 进行索引操作，可以对选定的行进行进一步的操作或选择。例如，data["类别"][data["类别"] == "paper"] 表示选择 "类别" 列中值为 "paper" 的行。
    data["类别"][data["类别"] == "paper"] = 0
    data["类别"][data["类别"] == "scissors"] = 1
    data["类别"][data["类别"] == "rock"] = 2
    x_data = data.iloc[:, :-1].to_numpy(np.float32)
    y_data = data["类别"].to_numpy(dtype=np.float32)
    # 分割训练集和测试集
    train_data, test_data, train_label, test_label = model_selection.train_test_split(
        x_data, y_data, random_state=1, test_size=0.25, train_size=0.75)
    # 定义SVM分类器
    svmClassfication = svm.SVC(
        C=100, kernel="rbf", gamma=1e-5, decision_function_shape="ovr", max_iter=50000)
    # 训练SVM分类器
    svmClassfication = svmClassfication.fit(train_data, train_label)
    print("训练集:{: >8.4f}%".format(
        svmClassfication.score(train_data, train_label)*100))
    print("验证集:{: >8.4f}%".format(
        svmClassfication.score(test_data, test_label)*100))
    # 保存分类器
    joblib.dump(svmClassfication, 'D:\\desktop\\opencvcode\\teamquiz2.0\\assignment02\\savedfilesvm.pkl')

    
