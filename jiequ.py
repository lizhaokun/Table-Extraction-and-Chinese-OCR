import cv2
import os

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

def main():
    path = 'img'
    img_paths, filenames = get_img(path)
    for img_path, filename in zip(img_paths, filenames):
        img = cv2.imread(img_path)
        img_jiequ = img[0 : img.shape[0] // 3, 0 : img.shape[1] // 2, :]
        cv2.imwrite('jiequ/' + filename, img_jiequ)


if __name__ == '__main__':
    main()