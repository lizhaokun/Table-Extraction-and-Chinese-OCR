import os
from PIL import Image
from Extract_Rotate import Rotation_Correct
from formatcut import *

def get_img(input_Path):
    img_paths = []
    filenames = []
    for (path, dirs, files) in os.walk(input_Path):
        for filename in files:
            ext = ('.jpg','.png','.tif','.jpeg')
            if filename.endswith(tuple(ext)):
                img_paths.append(path+'/'+filename)
                filenames.append(filename)
    return img_paths, filenames



def Resize_img(img_path,resize_temp,Size=1800):
    if not os.path.exists(resize_temp):
        os.mkdir(resize_temp)
    try:
        img = Image.open(img_path)
        img_name = resize_temp + '/' + img_path.split('/')[-1][:-4]+'.jpg'
        if img.size[0]>Size or img.size[1]>Size:
            img.thumbnail((Size, Size))
            print(img.format, img.size, img.mode)
            img.save(img_name, 'JPEG')
        else:
            img.save(img_name, 'JPEG')
            print('img isnot need deal,copy...')
    except:
        print('Image read fail')



def main():
    # 1. 图片大小归一化，并保存
    # img_path = 'img'
    resize_temp ='data/test'
    # csv_path = 'data/csv'

    # 2. 尝试对图像进行旋转、仿射、gamma校正
    img_paths, filenames = get_img(resize_temp)
    for img_path, filename in zip(img_paths, filenames):
        print('doing.....',img_path)
        # result_path =csv_path +'/'+img_path.split('/')[-1][:-4]+'.csv'
        # image, mask = Remove_watermark(img_path)
        image = cv2.imread(img_path)
        Rotation_img = Rotation_Correct(image, MinLineLength=100, MaxLineGap=20)


    # 3. 提取表格目标区域进行识别
        Result = []
        src_img, mask = find_Table_Contours(Rotation_img)
        print('开始识别：。。。。。。。。。。。')
        Get_Roi_Area(src_img, mask, filename)
        print('识别完成。。。。。。。。。。。。')
        # ImgAll_hang = Form_Cutting(temp_Img,temp_mask)

        # for i in range(len(ImgAll_hang)):
        #     result = []
        #     for j in range(len(ImgAll_hang[i])):
        #         time.sleep(0.2)
        #         text = pytesseract.image_to_string(ImgAll_hang[i][j], lang='chi_sim')  # 使用简体中文解析图片
        #         print('text:',text)
        #         result.append(text)
        #     Result.append(result)
        # with open(result_path, 'w', newline='', encoding='utf-8') as csv_file:
        #     field = ('field1', 'field2', 'field3', 'field4', 'field5', 'field6')
        #     csv_writer = csv.writer(csv_file)
        #     csv_writer.writerow(field)
        #     csv_writer.writerows(Result)


if __name__ == '__main__':
    main()