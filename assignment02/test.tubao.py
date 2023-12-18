
import cv2
# --------------读取并绘制原始图像------------------
o = cv2.imread('hand.bmp')  
cv2.imshow("original",o)
# --------------提取轮廓------------------
gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)  
contours, hierarchy = cv2.findContours(binary,
                                             cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_SIMPLE)  
# --------------寻找凸包，得到凸包的角点------------------
hull = cv2.convexHull(contours[0])
# --------------绘制凸包------------------
cv2.polylines(o, [hull], True, (0, 255, 0), 2)
"""cv2.polylines() 函数的参数如下所示：

o：要绘制多边形线段的图像。
[hull]：包含多边形顶点坐标的列表。在这里，hull 是由 cv2.convexHull() 函数计算得到的凸包顶点坐标。
True：表示绘制闭合的多边形，即连接首尾两个顶点形成一个封闭的轮廓。
(0, 255, 0)：线段的颜色，这里是绿色，表示 (R, G, B) 的颜色值。
2：线段的宽度。"""
# --------------显示凸包------------------
cv2.imshow("result",o)
cv2.waitKey()
cv2.destroyAllWindows()
