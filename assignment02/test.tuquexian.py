"""凸包缺陷"""

# https://blog.csdn.net/qq_40784418/article/details/105999175 凸包检测与缺陷

import cv2
#----------------原图--------------------------
img = cv2.imread('hand.bmp')
cv2.imshow('original',img)
#----------------构造轮廓--------------------------
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 127, 255,0)
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_TREE,
                                 
            cv2.CHAIN_APPROX_SIMPLE)  
#----------------凸包--------------------------
"""处理了第一个轮廓，因为它是最大的轮廓。通常，在处理图像时，我们只对最大的轮廓感兴趣"""
cnt = contours[0]
hull = cv2.convexHull(cnt,returnPoints = False)#这里需要False
defects = cv2.convexityDefects(cnt,hull)
print("defects=\n",defects)
#----------------构造凸缺陷--------------------------
for i in range(defects.shape[0]):#i 是一个循环变量，它的取值是从 0 到 defects.shape[0]-1 的整数序列中的每一个元素
    #shape【0】是行数
    s,e,f,d = defects[i,0]#起点 终点 最远的点 最远近似距离
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img,start,end,[0,0,255],2)
    cv2.circle(img,far,5,[255,0,0],-1)#以最远点为圆心，绘制半径为5的圆，-1代表将圆完全填冲充
    # print(cnt[f]) cnt【f】： [[152 152]]
    
#----------------显示结果，释放图像--------------------------
cv2.imshow('result',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

