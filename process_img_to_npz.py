# -*- coding: utf-8 -*-
# @Author: biying
# @Date:   2017-06-30 10:20:35
# @Last Modified by:   Anderson
# @Last Modified time: 2018-05-29 13:31:03

import os
import numpy as np
import matplotlib.image as mpimg
# from PIL import Image
from time import time
import math

CHUNK_SIZE = 256

def process_img(img_path, key):
	print(img_path, key)

	# Use PIL to convert image file to numpy array
	# image = Image.open(img_path)
	# image_array = np.array(image)
	# image_array = np.expand_dims(image_array,axis = 0)
	
	# Use matplotlib to convert image file to numpy array
	image_array = mpimg.imread(img_path)
	image_array = np.expand_dims(image_array,axis = 0)
	print(image_array.shape)
	
	if key == 2:
		label_array = [ 0.,  0.,  1.,  0.,  0.]
	elif key ==3:
		label_array = [ 0.,  0.,  0.,  1.,  0.]
	elif key == 0:
		label_array = [ 1.,  0.,  0.,  0.,  0.]
	elif key == 1:
		label_array = [ 0.,  1.,  0.,  0.,  0.]
	elif key == 4:
		label_array = [ 0.,  0.,  0.,  0.,  1.]

	return (image_array, label_array)

if __name__ == '__main__':
	path = "training_data"
	files= os.listdir(path)
	turns = int(math.ceil(len(files) / CHUNK_SIZE))
	print("number of files: {}".format(len(files)))
	print("turns: {}".format(turns))

	for turn in range(0, turns):
		train_labels = np.zeros((1,5),'float')
		train_imgs = np.zeros([1,120,160,3])

		CHUNK_files = files[turn*CHUNK_SIZE: (turn+1)*CHUNK_SIZE]
		print("number of CHUNK files: {}".format(len(CHUNK_files)))
		for file in CHUNK_files:
			if not os.path.isdir(file) and file[len(file)-3:len(file)] == 'jpg': 
				try:
					key = int(file[0])
					image_array, label_array = process_img(path+"/"+file, key)
					train_imgs = np.vstack((train_imgs, image_array))
					train_labels = np.vstack((train_labels, label_array))
				except:
					print('prcess error')

		# 去掉第0位的全零图像数组，全零图像数组是 train_imgs = np.zeros([1,120,160,3]) 初始化生成的
		train_imgs = train_imgs[1:, :]               # 从第一位开始取，因为第0位是初始化的
		train_labels = train_labels[1:, :]
		file_name = str(int(time()))                 # 文件名直接取时间
		directory = "training_data_npz"              # 文件夹

		if not os.path.exists(directory):
			os.makedirs(directory)
		try:    
			np.savez(directory + '/' + file_name + '.npz', train_imgs=train_imgs, train_labels=train_labels)
		except IOError as e:
			print(e)
