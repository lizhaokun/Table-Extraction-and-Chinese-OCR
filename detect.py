import cv2
import argparse
import numpy as np
from dnn.main import text_ocr
from config import scale_d,maxScale,TEXT_LINE_SCORE

def job(path):
    img = cv2.imread(path)
    image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    data = text_ocr(image, scale_d, maxScale, TEXT_LINE_SCORE)
    res = {'data':data,'errCode':0}
    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="", help='path of image')
    opt = parser.parse_args()
    job(opt.path)