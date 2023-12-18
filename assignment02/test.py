import os
import warnings
import cv2 as cv
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from scan_gesture import *
import cv2
import math
from PIL import ImageFont, ImageDraw, Image
# 手势识别函数
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)    
#==============主程序======================
while(cap.isOpened()):
    ret,frame = cap.read() # 读取摄像头图像
#     print(frame.shape)   #获取窗口大小
    frame = cv2.flip(frame,1)   #沿着y轴转换下方向,变成与自己方向一致的镜像
    #===============设定一个固定区域作为识别区域=============
    roi = frame[20:300,310:580] # 将右上角设置为固定识别区域
    cv2.rectangle(frame,(325,70),(580,330),(0,0,255),0) # 将选定的区域标记出来,让圈出来的变成自己测试过的，实际检测范围更大一些
    #===========在hsv色彩空间内检测出皮肤===============
    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)    #色彩空间转换
    lower_skin = np.array([0,28,70],dtype=np.uint8)   #设定范围，下限
    upper_skin = np.array([20, 255, 255],dtype=np.uint8)  #设定范围，上限
    mask = cv2.inRange(hsv,lower_skin,upper_skin)   #确定手所在区域
    #===========预处理===============
    kernel = np.ones((2,2),np.uint8)   #构造一个核
    mask = cv2.dilate(mask,kernel,iterations=4)   #膨胀操作
    mask = cv2.GaussianBlur(mask,(5,5),100)       #高斯滤波    
    #=================找出轮廓===============
    #查找所有轮廓
    contours,h = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #从所有轮廓中找到最大的，作为手势的轮廓
    cnt = max(contours,key=lambda x:cv2.contourArea(x)) #max函数通过后面的自定义函数来在contours选出面积最大的轮廓为cnt 
    areacnt = cv2.contourArea(cnt)   #获取轮廓面积
    #===========获取轮廓的凸包=============
    hull = cv2.convexHull(cnt)   #获取轮廓的凸包,用于计算面积，返回坐标
    # hull = cv2.convexHull(cnt,returnPoints=False)
    areahull = cv2.contourArea(hull)   #获取凸包的面积
    #===========获取轮廓面积、凸包的面积比=============
    arearatio = areacnt/areahull   
    # 轮廓面积/凸包面积 ：
    # 大于0.9，表示几乎一致，是手势0
    # 否则，说明凸缺陷较大，是手势1.
    #===========获取凸缺陷=============
    hull = cv2.convexHull(cnt,returnPoints=False) #使用索引，returnPoints=False
    defects = cv2.convexityDefects(cnt,hull)    #获取凸缺陷
    #===========凸缺陷处理==================
    n=0 #定义凹凸点个数初始值为0 
    #-------------遍历凸缺陷，判断是否为指间凸缺陷--------------
    for i in range(defects.shape[0]):##i 是一个循环变量，它的取值是从 0 到 defects.shape[0]-1 的整数序列中的每一个元素
        s,e,f,d, = defects[i,0]
        start = tuple(cnt[s][0])#print(cnt[f]) cnt【f】： [[152 152]]
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
        #--------计算手指之间的角度----------------
        angle = math.acos((b**2 + c**2 -a**2)/(2*b*c))*57
        #-----------绘制手指间的凸包最远点-------------
        #角度在[20,90]之间的认为是不同手指所构成的凸缺陷
        if angle<=90 and d>20:
            n+=1
            cv2.circle(roi,far,3,[255,0,0],-1)   #用蓝色绘制最远点
        #----------绘制手势的凸包--------------
        cv2.line(roi,start,end,[0,255,0],2) 
    #============通过凸缺陷个数及面积比判断识别结果=================
    if n==0:           #0个凸缺陷，可能为0，也可能为1
        if arearatio>0.9:     #轮廓面积/凸包面积>0.9，判定为拳头，识别为0
            result='0'
        else:
            result='1'   #轮廓面积/凸包面积<=0.9,说明存在很大的凸缺陷，识别为1
    elif n==1:        #1个凸缺陷，对应2根手指，识别为2
        result='2'
    elif n==2:        #2个凸缺陷，对应3根手指，识别为3
        result='3'
    elif n==3:        #3个凸缺陷，对应4根手指，识别为4
        result='4'
    elif n==4:        #4个凸缺陷，对应5根手指，识别为5
        result='5'
    if result=='0':
        gesture="rock"
    elif result=='2':
        gesture="scissor"
    elif result=='5':
        gesture="Paper"
    else:
        gesture="bu wan gun()"
   

    #============设置与显示识别结果相关的参数=================
    org=(400,80)
    org2=(350,400)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font2=cv2.FONT_HERSHEY_PLAIN
    fontScale=2
    color=(0,0,255)
    color2=(255,0,0)
    thickness=3
    #================显示识别结果===========================
    cv2.putText(frame,result,org,font,fontScale,color,thickness)
    cv2.putText(frame,gesture,org2,font,1,color,1)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(25)& 0xff  
    if k == 27:     # 键盘Esc键退出
        break
cv2.destroyAllWindows()
cap.release()
# warnings.filterwarnings("ignore")

# if __name__ == "__main__":
#     file_path = "D:\\desktop\\opencvcode\\teamquiz2.0\\assignment02\\data"
#     files = os.listdir(file_path)
#     # 加载训练好的SVM分类器
#     svmClassfication: svm = joblib.load('D:\\desktop\\opencvcode\\teamquiz2.0\\assignment02\\svm.pkl\\savedfilesvm.pkl')
#     cnt = 0
#     print("分类错误样本:")
#     for i in range(len(files)):
#         name = files[i]
#         file_name = file_path+"/"+name
#         # 分类文件名中的标签信息
#         label = name.split("_")[1]
#         # 转换为索引以便于比较
#         label_idx = get_idx(label)
#         # 读取图像并提取特征
#         src = cv.imread(file_name)
#         a, s, e, n, h = get_features(src)
#         x = np.array([a, s, e, n])
#         x = np.hstack([x, h]).reshape(1, -1)
#         # 获取预测值(标签索引)
#         y = svmClassfication.predict(x)
#         # 判断预测的准确性并计算准确率
#         if label_idx == y.item():
#             cnt += 1
#         else:
#             print("样本{}:标签{},被错分为:{}".format(i+1, label, get_label(y.item())))
#     if cnt == len(files):
#         print("无")
#     print(f"\n分类准确率:{cnt/len(files)*100}%")
#     joblib.dump(svmClassfication, 'D:\\desktop\\opencvcode\\teamquiz2.0\\assignment02\\svm.pkl\\savedfilesvm.pkl')
