
import cv2 as cv
import numpy as np
import matplotlib as plt
import pandas as pd
import os
import skimage.feature as sf
'''可能需要调thre二值化的参数'''
''''离心率的函数那里有问题，觉得for循环那里有语义错误'''

'''最后需要两个进行标签与下标互逆转换的函数不懂'''
def thre(src):
    '''二值图像获取'''
    
    # OTSU阈值分割
    t, img = cv.threshold(src, 20, 255, cv.THRESH_OTSU)
    # 对二值图像进行去噪操作
    #先腐蚀再膨胀，开运算
    #
    kernel = np.ones((3, 3), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=3)
    # 中值滤波
    img = cv.medianBlur(img, 7)
    return img

def circle_similar(src):
    '''似圆度'''
    contour, hierarchy = cv.findContours(
        src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)#src一般为灰度化图像或二值化图像 轮廓的检索方式：只检测外轮廓。轮廓逼近方法：储存所有的轮廓点

    length = 0
    area = 0
    # 寻找最大图形的周长和面积，来计算轮廓紧凑程度
    for i in range(len(contour)):
        c = contour[i].astype(np.float32)
        length = max(length, cv.arcLength(c, True))#c++有类似的算法 c=max（c，t）
        area = max(area, cv.contourArea(c))
    return length**2 / area#代表轮廓的紧凑特征
def Area(src):
    '''面积'''
    contour, hierarchy = cv.findContours(
        src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    area = 0
    # 寻找最大图形的面积
    for i in range(len(contour)):
        c = contour[i].astype(np.float32)
        area = max(area, cv.contourArea(c))
    return area#理论同上面函数
def hr(src):
    '''角点数量'''
    # Harris角点检测
    points = cv.cornerHarris(src, 3, 7, 0.05)
    # 统计非0点数量
    ret = np.count_nonzero(points)#计算有多少个角度变化的点
    return ret

def eccentricity(src):
    '''离心率'''
    contours, hierarchy = cv.findContours(
        src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    nmax = 0
    # 寻找最大轮廓
    for i in range(len(contours)):
        if len(contours[i]) > len(contours[nmax]):
            nmax = i
    # 计算外接椭圆
    ell = cv.fitEllipse(contours[nmax])
    a, b = max(ell[1])/2, min(ell[1])/2
    c = np.sqrt(a**2-b**2)
    return c/a

def hog(src):
    '''HOG'''
    # 降低图形分辨率以减小特征维度
    img = cv.resize(src, (150, 150))
    hog_array = sf.hog(img, orientations=8, pixels_per_cell=(
        15, 15), cells_per_block=(10, 10))#8是将频率分布直方图将360分成八份，每一份45°然后将
    return hog_array
def get_features(src):
    '''获取特征向量'''
    # 转换颜色空间
    img = cv.cvtColor(src, cv.COLOR_RGB2HSV)
    h_channel, s_channel, v_channel = cv.split(img)
    # 对h通道进行阈值分割
    img = thre(h_channel)#上面用的阈值分割
    # 获取特征
    a = Area(img)
    s = circle_similar(img)
    e = eccentricity(img)
    n = hr(img)
    h = hog(img)
    return a, s, e, n, h

def get_label(idx):
    '''根据索引获取标签内容'''
    labels = ["paper", "scissors", "rock"]
    return labels[int(idx)]


def get_idx(label):
    '''根据标签获取对应索引'''
    labels = {"paper": 0, "scissors": 1, "rock": 2}
    return labels[label]
