# -*- coding: UTF-8 -*-
# import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
# from dnn.main import text_ocr
# from config import scale_d,maxScale,TEXT_LINE_SCORE

def find_Table_Contours(src_img):
    src_img0 = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_img0 = cv2.GaussianBlur(src_img0, (3, 3), 0)
    src_img1 = cv2.bitwise_not(src_img0)
    AdaptiveThreshold = cv2.adaptiveThreshold(src_img1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 15,-2)
    horizontal = AdaptiveThreshold.copy()
    vertical = AdaptiveThreshold.copy()
    scale = 20

    # 水平线
    horizontalSize = int(horizontal.shape[1] / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalSize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)

    # 竖直线
    verticalsize = int(vertical.shape[0] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    mask = horizontal + vertical
    return src_img,mask


def Get_Roi_Area(src_img,mask, filename):
    height, width, = src_img.shape[:2]
    image_copy = src_img.copy()
    roi_mask = mask
    #plt.imshow(image_copy)
    #plt.show()
    #plt.imshow(roi_mask)
    #plt.show()
    _, cnts_canny, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts_canny) > 0:
        kuang = []
        for i, c in enumerate(cnts_canny):
            x, y, w, h = cv2.boundingRect(c)
            kuang.append((w * h, c))
        kuang = sorted(kuang, key=lambda s: s[0], reverse=True)

        if kuang[0][0] < height * width:
            Max_contour = kuang[0][1]
        else:
            Max_contour = kuang[1][1]
        x, y, w, h = cv2.boundingRect(Max_contour)
        dealt = int(min(height, width) / 26)
        # print(image_copy.shape[:2])
        temp_Img = image_copy[y-dealt:y + h + 1, x-dealt:x + w + 1]
        temp_Img_copy = temp_Img
        temp_mask = mask[y-dealt:y + h + 1, x-dealt:x + w + 1]
        # print(temp_mask.shape)
        temp_mask[:, (temp_mask.shape[1] - 5): (temp_mask.shape[1] - 2)] = 255
        temp_mask[(temp_mask.shape[0] - 5): (temp_mask.shape[0] - 2), :] = 255
        # cv2.imwrite('temp/cut_' + filename + '.jpg', temp_Img)
        # cv2.imwrite('temp/cut_mask_' + filename + '.jpg', temp_mask)
        #plt.imshow(temp_Img)
        #plt.show()
        #plt.imshow(temp_mask)
        #plt.show()

        rows, cols = temp_mask.shape
        scale = 40
        # 识别横线
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
        eroded = cv2.erode(temp_mask, kernel, iterations=1)#腐蚀
        # cv2.imshow("Eroded Image",eroded)
        dilatedcol = cv2.dilate(eroded, kernel, iterations=1)#膨胀

        # 识别竖线
        scale = 20
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
        eroded = cv2.erode(temp_mask, kernel, iterations=1)
        dilatedrow = cv2.dilate(eroded, kernel, iterations=1)

        bitwiseAnd = cv2.bitwise_and(dilatedcol, dilatedrow)

        mask, contours, hierarchy = cv2.findContours(temp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        # print(length)
        # print(hierarchy)

        small_rects = []
        big_rects = []
        for i in range(length):
            cnt = contours[length - 1 - i]
            area = cv2.contourArea(cnt)
            if area < 10:
                continue
            approx = cv2.approxPolyDP(cnt, 3, True)  # 3
            x, y, w, h = cv2.boundingRect(approx)
            rect = (x, y, w, h)
            # cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            roi = bitwiseAnd[y:y + h, x:x + w]
            # plt.imshow(roi)
            # plt.show()
            roi, bitwiseAnd_contours, joints_hierarchy = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print (len(bitwiseAnd_contours))
            # if h < 80 and h > 20 and w > 10 and len(joints_contours)<=4:
            if h < 100 and h > 20 and w > 10 and len(bitwiseAnd_contours) <= 6:  # important
                # cv2.rectangle(temp_Img, (x, y), (x + w, y + h), (255 - h * 3, h * 3, 0), 2)
                small_rects.append(rect)
            #print(len(small_rects))
        plt.imshow(temp_Img)
        plt.show()

        num = 0
        for j in small_rects:
            x_j, y_j, w_j, h_j = j
            cut_region = temp_Img_copy[y_j: y_j + h_j, x_j: x_j + w_j, :]
            cut_region_data = cv2.cvtColor(cut_region, cv2.COLOR_RGB2BGR)
            plt.imshow(cut_region_data)
            plt.show()
            cv2.imwrite('data/temp/cut_' + str(num) + '.jpg', cut_region_data)
            num += 1
            # data = text_ocr(cut_region_data, scale_d, maxScale, TEXT_LINE_SCORE)
            # print(data)

        # cut_regions = sorted(small_rects, key=lambda s : s[3], reverse=True)
        # for i in cut_regions:
        #     x_i, y_i, w_i, h_i = i
        #     cut_region = temp_Img[y_i : y_i + h_i, x_i : x_i + w_i, :]
        #     cut_region_data = cv2.cvtColor(cut_region, cv2.COLOR_RGB2BGR)
        #     plt.imshow(cut_region_data)
        #     plt.show()
            # data = text_ocr(cut_region_data, scale, maxScale, TEXT_LINE_SCORE)
            # print(data)
            # plt.imshow(cut_region)
            # plt.show()
            # plt.imshow(cut_region_data)
            # plt.show()
        # cv2.imshow('bounging', temp_Img)
        # cv2.waitKey(0)

        # cv2.imshow('表格交点显示', bitwiseAnd)
        # cv2.waitKey(0)
        # # 识别黑白图中的白色交叉点，将横纵坐标取出
        # ys, xs = np.where(bitwiseAnd > 0)
        #
        # mylisty = []  # 纵坐标
        # mylistx = []  # 横坐标
        #
        # # 通过排序，获取跳变的x和y的值，说明是交点，否则交点会有好多像素值值相近，我只取相近值的最后一点
        # # 这个10的跳变不是固定的，根据不同的图片会有微调，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
        # i = 0
        # myxs = np.sort(xs)
        # for i in range(len(myxs) - 1):
        #     if (myxs[i + 1] - myxs[i] > 20):
        #         mylistx.append(myxs[i])
        #     i = i + 1
        # mylistx.append(myxs[i])  # 要将最后一个点加入
        #
        # i = 0
        # myys = np.sort(ys)
        # # print(np.sort(ys))
        # for i in range(len(myys) - 1):
        #     if (myys[i + 1] - myys[i] > 20):
        #         mylisty.append(myys[i])
        #     i = i + 1
        # mylisty.append(myys[i])  # 要将最后一个点加入
        #
        # print('mylisty', mylisty)
        # print('mylistx', mylistx)
        #
        # # 循环y坐标，x坐标分割表格
        # for i in range(len(mylisty) - 1):
        #     for j in range(len(mylistx) - 1):
        #         # 在分割时，第一个参数为y坐标，第二个参数为x坐标
        #         ROI = temp_Img[mylisty[i] + 3:mylisty[i + 1] - 3, mylistx[j]:mylistx[j + 1] - 3]  # 减去3的原因是由于我缩小ROI范围
        #         cv2.imshow("分割后子图片展示：", ROI)
        #         cv2.waitKey(0)

        # cv2.imshow("表格竖线展示：", dilatedrow)
        # cv2.waitKey(0)
        # cv2.imshow("表格横线展示：", dilatedcol)
        # cv2.waitKey(0)
        # cv2.imshow('表格交点显示', bitwiseAnd)
        # cv2.waitKey(0)

        # return temp_Img, temp_mask


def __Lie_loacl__(temp_mask):
    height, width = temp_mask.shape[:2]
    scribeline = np.count_nonzero(temp_mask, axis=1)#掩模中每行非0元素的个数
    diffvalue = scribeline / width #每一行非0元素的数量在这一行中占的比例

    lie_local = np.argwhere(diffvalue > 0) #返回数组中值大于0的索引值
    lie_local = lie_local.reshape(1, -1)
    diff_local = np.diff(lie_local).tolist()[0]

    Local = []
    for i in range(len(diff_local)):
        if i == 0 and diff_local[i] < 10:
            Local.append(i)
        if i > 0 and diff_local[i] < 10 and 5<abs(diff_local[i - 1] -diff_local[i]):
            Local.append(i)
    lie_local = lie_local.tolist()[0]
    # print('__diff_local:',diff_local)
    # print('__hang_local:', lie_local)
    # print('__Local:', Local)
    return Local, lie_local


def __Hang_loacl__(cutHang):
    height, width = cutHang.shape[:2]
    scribeline = np.count_nonzero(cutHang, axis=0)
    print('||height:', height)
    diffvalue = scribeline / height

    hang_local = np.argwhere(diffvalue > 0)
    hang_local = hang_local.reshape(1, -1)
    diff_local = np.diff(hang_local).tolist()[0]

    Local = []
    for i in range(len(diff_local)):
        if i == 0 and diff_local[i] < 10:
            Local.append(i)
        if i > 0 and diff_local[i] < 10 and 5<abs(diff_local[i - 1] -diff_local[i]):
            Local.append(i)
    hang_local = hang_local.tolist()[0]
    # print('diff_local:',diff_local)
    # print('hang_local:', hang_local)
    # print('Local:', Local)
    return Local, hang_local


def Form_Cutting(temp_Img,temp_mask):
    Local, lie_local = __Lie_loacl__(temp_mask)
    masklie = []
    Imglie = []
    for i in range(len(Local)-1):
        y1,y2 = lie_local[Local[i]], lie_local[Local[i + 1]]
        print('|--|', y1, y2)
        masklie.append(temp_mask[y1:y2,:])
        Imglie.append(temp_Img[y1:y2, :])

    ImgAll_hang = []
    for i in range(len(masklie)):
        Img_hang = []
        Local, hang_local = __Hang_loacl__(masklie[i])
        for j in range(len(Local) - 1):
            x1, x2 = hang_local[Local[j]], hang_local[Local[j + 1]]
            print('__||__', x1, x2)
            Img_hang.append(Imglie[i][:, x1:x2])
        ImgAll_hang.append(Img_hang)
    return ImgAll_hang


# if __name__ == '__main__':
#     input_Path = 'img\\stamp.jpg'
#     image = cv2.imread(input_Path)
#     src_img, mask = find_Table_Contours(image)
#     Get_Roi_Area(src_img,mask)