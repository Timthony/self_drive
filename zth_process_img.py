# 将图片处理为npz格式

import os
import numpy as np
import matplotlib.image as mpimg
from time import time
import math
from PIL import Image

CHUNK_SIZE = 256




def process_img(img_path, key):

    print(img_path, key)
    image = Image.open(img_path)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)




    #image_array = mpimg.imread(img_path)
    #image_array = np.expand_dims(image_array, axis=0)

    print(image_array.shape)

    if key == 2:
        label_array = [0., 0., 1., 0., 0.]
    elif key == 3:
        label_array = [0., 0., 0., 1., 0.]
    elif key == 0:
        label_array = [1., 0., 0., 0., 0.]
    elif key == 1:
        label_array = [0., 1., 0., 0., 0.]
    elif key == 4:
        label_array = [0., 0., 0., 0., 1.]

    return (image_array, label_array)
    # 返回图片的数据（矩阵），和对应的标签值


if __name__ == '__main__':
    path = "training_data"
    files = os.listdir(path)                           # 将该路径下的文件名都存入列表
    turns = int(math.ceil(len(files) / CHUNK_SIZE))
    print("number of files: {}".format(len(files)))
    print("turns: {}".format(turns))

    for turn in range(0, turns):
        train_labels = np.zeros((1, 5), 'float')
        train_imgs = np.zeros([1, 120, 160, 3])

        CHUNK_files = files[turn * CHUNK_SIZE: (turn + 1) * CHUNK_SIZE]
        print("number of CHUNK files: {}".format(len(CHUNK_files)))
        for file in CHUNK_files:
            if not os.path.isdir(file) and file[len(file) - 3:len(file)] == 'jpg':
                try:
                    key = int(file[0])
                    image_array, label_array = process_img(path + "/" + file, key)
                    train_imgs = np.vstack((train_imgs, image_array))
                    train_labels = np.vstack((train_labels, label_array))
                except:
                    print('prcess error')

        # 去掉第0位的全零图像数组，全零图像数组是 train_imgs = np.zeros([1,120,160,3]) 初始化生成的
        train_imgs = train_imgs[1:, :]
        train_labels = train_labels[1:, :]
        file_name = str(int(time()))
        directory = "/Volumes/Seagate Expansion Drive/tianhangz/project/selfdrive/training_data_npz"

        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            np.savez(directory + '/' + file_name + '.npz', train_imgs=train_imgs, train_labels=train_labels)
        except IOError as e:
            print(e)


