# -*- coding: UTF-8 -*-
import cv2
import math
import numpy as np
from collections import defaultdict
#from matplotlib import pyplot as plt


def __Duplicate_elements__(array,axis=0):
    dd = defaultdict(list)
    temp_index = []
    temp_values = []
    temp_length = []
    Duplicates = []
    for i, v in enumerate(array):
        temp_index.append(i)
        temp_values.append(v)
    for k, val in zip(temp_values, temp_index):
        dd[k[axis]].append(val)
    for l in sorted(list(set([V[axis] for V in array]))):
        temp_length.append((l,len(dd[l])))
    temp_length = sorted(temp_length, key=lambda s: s[1], reverse=True)
    for m in array:
        if m[axis] in temp_length[0]:
            Duplicates.append(m)
    return Duplicates


def __four_point_transform__(img,locals):
    docCnt = locals
    widthA = np.sqrt(((locals[0][0] - locals[1][0]) ** 2) + ((locals[0][1] - locals[1][1]) ** 2))
    widthB = np.sqrt(((locals[2][0] - locals[3][0]) ** 2) + ((locals[2][1] - locals[3][1]) ** 2))
    maxWidth = min(int(widthA), int(widthB))+0.8*(max(int(widthA), int(widthB)) - min(int(widthA), int(widthB)))
    heightA = np.sqrt(((locals[1][0] - locals[2][0]) ** 2) + ((locals[1][1] - locals[2][1]) ** 2))
    heightB = np.sqrt(((locals[3][0] - locals[0][0]) ** 2) + ((locals[3][1] - locals[0][1]) ** 2))
    maxHeight = int(max(int(heightA), int(heightB))*1.1)

    print('docCnt:',docCnt)
    pts1 = np.float32([docCnt[0], docCnt[1], docCnt[2], docCnt[3]])
    pts2 = np.float32([[0, 0],[maxWidth,0],[maxWidth,maxHeight],[0,maxHeight]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (int(maxWidth),maxHeight))
    return dst



def __Getting_Affine_line__(src_img,LineSetShu, LineSet):
    H_row, W_cols = src_img.shape[:2]
    LineSet = sorted(LineSet, key=lambda s: s[1], reverse=False)
    LineSetShu = sorted(LineSetShu, key=lambda s: s[0], reverse=False)

    line_A = [A1 for A1 in LineSet if
              1 < A1[0] < W_cols
              and 1 < A1[2] < W_cols
              and 1 < A1[1] < 0.5*H_row
              and 1 < A1[3] < 0.5*H_row]

    line_C = [C1 for C1 in LineSet if
              1 < C1[0] < W_cols
              and 1 < C1[2] < W_cols
              and 0.5*H_row < C1[1] < H_row
              and 0.5*H_row < C1[3] < H_row]

    line_B = [B1 for B1 in LineSetShu if
              0.3 * W_cols < B1[0] <= W_cols
              and 0.3 * W_cols < B1[2] <= W_cols
              and 1 < B1[1] < H_row
              and 1 < B1[3] < H_row]

    line_D = [D1 for D1 in LineSetShu if
              1 < D1[0] < 0.7 * W_cols
              and 1 < D1[2] < 0.7 * W_cols
              and 1 < D1[1] < H_row
              and 1 < D1[3] < H_row]

    if len(line_A[0][:4]) < 4:
        line_A[0][:4] = [5, 5, 10, 5]
    if len(line_B[-1][:4]) < 4:
        line_B[-1][:4] = [W_cols-5, 5, W_cols-5, 10]
    if len(line_C[-1][:4]) < 4:
        line_C[-1][:4] = [5, H_row-5, 10, H_row-5]
    if len(line_D[0][:4]) < 4:
        line_D[0][:4] = [5, 5, 5, 10]

    x7, y7 = __Intersection__(line_A[0][:4], line_B[-1][:4])
    x8, y8 = __Intersection__(line_B[-1][:4], line_C[-1][:4])
    x9, y9 = __Intersection__(line_C[-1][:4], line_D[0][:4])
    x10,y10 = __Intersection__(line_D[0][:4], line_A[0][:4])
    Affine_coordinates = [[x10, y10], [x7, y7], [x8, y8], [x9, y9]]
    return Affine_coordinates


def __Intersection__(point1,point2):
    x1, y1, x2, y2 = point1
    x_1, y_1, x_2, y_2 = point2

    A1 = y2 - y1
    B1 = x1 - x2
    C1 = y1 * x2 - x1 * y2

    A2 = y_2 - y_1
    B2 = x_1 - x_2
    C2 = y_1 * x_2 - x_1 * y_2

    x = (B1*C2-B2*C1)/(A1*B2-A2*B1)
    y = (C1*A2-A1*C2)/(A1*B2-A2*B1)
    return int(x),int(y)


def gamma_transform(img_array, gamma=0.455):
    img0 = img_array
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img0, gamma_table)


def Affine_Correct(image,LineSetShu,LineSet):
    image_copy = image.copy()
    Affine_coordinates = __Getting_Affine_line__(image_copy, LineSetShu, LineSet)
    image_copy = __four_point_transform__(image_copy, np.array(Affine_coordinates))
    return image_copy


def nothing(poss):
    pass

def Rotation_Correct(image,MinLineLength=100,MaxLineGap=20):
    # image = cv2.imread(image)
    # try:
    #    image = gamma_transform(image)
    # except:
    #    pass
    rows, cols, = image.shape[:2]
    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_Gauss = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray_Gauss, 50, 150, apertureSize=3)
    # cv2.createTrackbar('Min', 'Canny', 0, 100, nothing)
    # cv2.createTrackbar('Max', 'Canny', 100, 200, nothing)
    # while True:
    #     cv2.imshow('Canny', edges)
    #     min = cv2.getTrackbarPos('Min', 'Canny')
    #     max = cv2.getTrackbarPos('Max', 'Canny')
    #     edges = cv2.Canny(gray_Gauss, min, max)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    #plt.imshow(edges)
    #plt.show()
    # cv2.namedWindow('edged', 2)
    # cv2.imshow('edged', edges)
    # cv2.waitKey(0)
    # cv2.destroyWindow('edged')

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 170, None, minLineLength=MinLineLength, maxLineGap=MaxLineGap)


    LineSet = []
    LineSetShu = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        Slope = float(y2 - y1) / float(x2 - x1 + 0.0001)  # 斜率 Slope
        Angle = math.atan(Slope)
        if abs(Angle) < np.pi/16 :
            lines_message = [x1, y1, x2, y2, Slope, round(Angle, 2)]
            LineSet.append(lines_message)
        elif abs(Angle) > 7*np.pi/16 :
            lines_message =[x1, y1, x2, y2, Slope, round(Angle, 2),abs(round(Angle, 2))]
            LineSetShu.append(lines_message)
    Duplicate_Angle = __Duplicate_elements__(LineSet, axis=5)             # 统计角度最多的值
    LineShu_Angle = sorted(LineSetShu, key=lambda s: s[6], reverse=False) # 升序排列

    # 判断是否需要旋转
    Rotation_Angle = Duplicate_Angle[0][5]
    if 0.02 < abs(Rotation_Angle):
        M = cv2.getRotationMatrix2D((cols // 2, rows // 2), math.degrees(Rotation_Angle), 1)
        image_copy = cv2.warpAffine(image_copy, M, (cols, rows), borderValue=(189,187,179))
        print('rotateAngle:', Rotation_Angle)

    # 判断是否需要仿射变换
    if LineShu_Angle[0][6]<1.50:
        print('LineShu[0]:', LineShu_Angle[0][6])
        image_copy = Affine_Correct(image_copy,LineSetShu,LineSet)
    return image_copy



# if __name__ == '__main__':
#     img_path = 'img\\JDCXSFP.png'
#     Rotation_img = Rotation_Correct(img_path, MinLineLength=100, MaxLineGap=20)