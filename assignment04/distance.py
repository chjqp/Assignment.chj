
#!/usr/bin/env python
import cv2
import numpy as np
import math
#   642.0942  419.8089
#   641.3379  307.9839
#   460.9779  421.5323
#   460.3078  308.0075
# 0   0
# 0   50
# 80	10
# 80	50
# 自己的图像
# 3s
# 0 0
# 0 1071
# 1071 0
# 1071 1071

# 779.8810  485.0207
# 781.0772  568.9719
# 867.3904  483.1395
# 869.1147  567.1812
#世界坐标系 中间定位原点 物体实际大小 单位mm
point3s = np.array(([40,-25,0],[40,25,0],[-40,25,0],[-40,-25,0]),dtype=np.double)
# 像素坐标系。可以用ps识别 就是四点的行列坐标，四个点顺序要跟世界坐标对应
point2s = np.array(([642.0942,419.8089],[641.3379,307.9839],[460.3078,308.0075],[460.9779,421.5323],),dtype=np.double)
# point3s=np.array(([0, 0, 0],[0, 1071, 0],[1071,0, 0],[1071,1071,0]),dtype=np.double)
# point2s=np.array(([779.8118,485.1100],[781.0983,568.9147],[867.0161,483.2972],[868.8392,567.2364]),dtype=np.double)
# 相机内参矩阵 matlab求的需要转制 0
camera = np.array([[652.90522946, 0., 321.31072448], [0., 653.00569689, 255.77694781], [0., 0., 1.]],
                    dtype=np.double)
# 1592.31745225552	0	0
# 57.5714671473469	1463.09510100772	0
# 1047.46009281314	310.456147316488	1
# camera=np.array(([ 1592.31745225552, 57.5714671473469, 1047.46009281314],[0,1463.09510100772, 310.456147316488],[0,0,1]),dtype=np.double)
# 相机畸变系数 k1 k2 k3 p1 p2
dist = np.array([[-0.23397356, 0.39554466, 0.01463653, -0.00176085, -0.48223752]], dtype=np.double)
# dist=np.array(([-0.265646657440131,1.33169555991469,0, 0.132817697344621,-0.00923216493667286]),dtype=np.double)
#dist=dist.T
#dist=np.zeros((5,1))
# found,r,t=cv2.solvePnP(point3s,point2s,camera,dist,flags=cv2.SOLVEPNP_SQPNP) #计算雷达相机外参,r-旋转向量，t-平移向量
found,r,t=cv2.solvePnP(point3s,point2s,camera,dist) #计算雷达相机外参,r-旋转向量，t-平移向量
R=cv2.Rodrigues(r)[0] #旋转向量转旋转矩阵

# -0.9982 0.0056 - 0.0589
# -0.0056 - 1.0000 - 0.0004
# -0.0589 - 0.0001  0.9983
# R = np.array(([-0.9982,-0.0056,-0.0589],[0.0056,-1.0000,-0.0001],[-0.0589,-0.0589,0.9983]))
# t = np.array(([49.7671288089843],[20.5618256699387],[375.298657782898]))

#
    # 0.9999    0.0081   -0.0106
   # -0.0067    0.9919    0.1267
   #  0.0116   -0.1266    0.9919
# R = np.array(([0.9999,-0.0067,0.0116],[0.0081,0.9919,-0.1266],[-0.0106,0.1267,0.9919]))
# -3340.24759865471	2212.59138840833	19889.3443428461
# t = np.array(([-3340.24759865471],[2212.59138840833],[19889.3443428461]))
camera_position=-np.matrix(R).T*np.matrix(t) #相机位置

print(camera_position) #camera_position[2]就是距离

d3=np.array([[-3.14925, -1.54094, -1.06652]])
d2,_=cv2.projectPoints(d3,r,t,camera,dist)#重投影验证
print(r)
print(t)
print(d2)
