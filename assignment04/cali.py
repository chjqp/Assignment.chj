
import cv2
import numpy as np
import glob

# 定义棋盘格的尺寸
# CHECKERBOARD = (6, 9)
CHECKERBOARD=(11,6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

"""达到所需的精度（cv2.TERM_CRITERIA_EPS）
迭代次数达到30次（cv2.TERM_CRITERIA_MAX_ITER）
第三个参数 0.001 是用于 cv2.TERM_CRITERIA_EPS 的精度阈值。"""

# 创建向量以存储每张棋盘格图像的3D点
objpoints = []
# 创建向量以存储每张棋盘格图像的2D点
imgpoints = []

# 定义3D点的世界坐标

objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
"""第一个值 1 表示数组长度
第二个值 CHECKERBOARD[0]*CHECKERBOARD[1] 表示数组宽度，即棋盘格上角点的总数。
第三个值 3 表示数组每个元素的维度，也就是数组中每个元素的大小。"""
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)#mgird生成棋盘行数和列数的网格，
#T.reshape 用来对矩阵转置，并重新塑造大小为多行两列的形式
#objp的第一个元素，所有行，前两列
prev_img_shape = None

# 获取给定目录中存储的每个图像的路径
images = glob.glob('D:\desktop\opencvcode\\teamquiz2.0\\assignment04\image'+'\\*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 寻找棋盘格角点
    # 如果在图像中找到了所需数量的角点，则ret为True
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    # print(ret)

    """
    如果检测到了所需数量的角点，
    我们会对像素坐标进行精细调整，并在棋盘格图像上显示它们
    """
    if ret == True:
        objpoints.append(objp)
        # 对给定的2D点进行像素坐标的精细调整。
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # 绘制并显示角点
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    cv2.imshow('img',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

"""
通过传递已知的3D点（objpoints）
和检测到的角点的相应像素坐标（imgpoints）
进行相机校准
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("相机矩阵：\n")
print(mtx)
print("畸变系数：\n")
print(dist)
print("旋转向量：\n")
print(rvecs)
print("平移向量：\n")
print(tvecs)
"""以一个点为原点，然后给距离"""
l =15.0
point3s = np.empty(shape=(0, 3), dtype=np.double)
for i in range(6):
    for j in range(11):
        point = np.array([i*l, j*l, 0.0], dtype=np.double)
        point3s = np.append(point3s, [point], axis=0)

# 像素坐标系。可以用ps识别 就是四点的行列坐标，四个点顺序要跟世界坐标对应
point2s = corners2#np.array(([642.0942,419.8089],[641.3379,307.9839],[460.3078,308.0075],[460.9779,421.5323],),dtype=np.double)
# point3s=np.array(([0, 0, 0],[0, 1071, 0],[1071,0, 0],[1071,1071,0]),dtype=np.double)
# point2s=np.array(([779.8118,485.1100],[781.0983,568.9147],[867.0161,483.2972],[868.8392,567.2364]),dtype=np.double)
# 相机内参矩阵 matlab求的需要转制 0
camera = np.array([[1059.03451, 0.0, 631.98157], 
                   [0.0, 1061.10015, 455.66343], 
                   [0.0, 0.0, 1.0]], dtype=np.double)

dist = np.array([[0.27264704, -0.3168559, 0.04941088, 0.00261159, 0.58961178]], dtype=np.double)

#dist=dist.T
#dist=np.zeros((5,1))
# found,r,t=cv2.solvePnP(point3s,point2s,camera,dist,flags=cv2.SOLVEPNP_SQPNP) #计算雷达相机外参,r-旋转向量，t-平移向量
found,r,t=cv2.solvePnP(point3s,point2s,camera,dist) #计算雷达相机外参,r-旋转向量，t-平移向量
R=cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵
camera_position=-np.matrix(R).T*np.matrix(t) #相机位置

# print(camera_position) #camera_position[2]就是距离

d3=np.array([[-3.14925, -1.54094, -1.06652]])
d2,_=cv2.projectPoints(d3,r,t,camera,dist)#重投影验证
# print(r)
# print(t)
print(f"距离是{camera_position[2]}")


