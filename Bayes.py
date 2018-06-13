#encoding=utf-8


import numpy as np
import cv2
from PIL import Image
import time

from sklearn.metrics import accuracy_score


#二值化
def binaryzation(img):
	cv_img = img.astype(np.uint8)
	cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY_INV,cv_img)
	return cv_img

def Train(trainset,train_labels):
	prior_probability = np.zeros(class_num)                         # 先验概率
	conditional_probability = np.zeros((class_num,feature_len,2))   # 条件概率

	# 计算先验概率及条件概率
	for i in range(len(train_labels)):


		img = binaryzation(trainset[i])     # 图片二值化

		#print(img.shape)
		label = train_labels[i]

		prior_probability[label] += 1

		for j in range(feature_len):
			conditional_probability[label][j][img[j]] += 1

	# 将概率归到[1.10001]
	for i in range(class_num):
		for j in range(feature_len):

			# 经过二值化后图像只有0，1两种取值
			pix_0 = conditional_probability[i][j][0]
			pix_1 = conditional_probability[i][j][1]

			# 计算0，1像素点对应的条件概率
			probalility_0 = (float(pix_0)/float(pix_0+pix_1))*100000 + 1
			probalility_1 = (float(pix_1)/float(pix_0+pix_1))*100000 + 1

			conditional_probability[i][j][0] = probalility_0
			conditional_probability[i][j][1] = probalility_1

	return prior_probability,conditional_probability

# 计算概率
def calculate_probability(img,label):
	probability = int(prior_probability[label])

	for i in range(len(img)):
		probability *= int(conditional_probability[label][i][img[i]])

	return probability

def Predict(testset,prior_probability,conditional_probability):
	predict = []

	for img in testset:
		#print(img)
		#print(img.shape)
		# 图像二值化
		img = binaryzation(img)
		#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素
		#img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

		max_label = 0
		max_probability = calculate_probability(img,0)

		for j in range(1,10):
			probability = calculate_probability(img,j)

			if max_probability < probability:
				max_label = j
				max_probability = probability


		predict.append(max_label)

	return np.array(predict)


class_num = 10
feature_len = 2025

if __name__ == '__main__':

	print ('Start read data')

	time_1 = time.time()


	train_features = np.fromfile("mnist_train_data",dtype=np.uint8)
	train_labels = np.fromfile("mnist_train_label",dtype=np.uint8)
	test_features = np.fromfile("mnist_test_data",dtype=np.uint8)
	test_labels = np.fromfile("mnist_test_label",dtype=np.uint8)


	train_features = train_features.reshape(60000,45,45)
	train_features = train_features.astype(np.float32)
	test_features = test_features.reshape(10000,45,45)
	test_features = test_features.astype(np.float32)
	train_labels = train_labels.astype(np.int32)
	test_labels = test_labels.astype(np.int32)

	train_features=train_features.flatten()
	train_features=train_features.reshape(60000,45*45)
	test_features=test_features.flatten()
	test_features=test_features.reshape(10000,45*45)

	#train_features = train_features[:1000]
	#test_features = test_features [:200]
	#train_labels =train_labels[:1000]
	#test_labels=test_labels[:200]



	time_2 = time.time()

	print ('read data cost ',time_2 - time_1,' second','\n')

	print ('Start training')
	prior_probability,conditional_probability = Train(train_features,train_labels)
	time_3 = time.time()
	print ('training cost ',time_3 - time_2,' second','\n')

	print ('Start predicting')
	test_predict = Predict(test_features,prior_probability,conditional_probability)
	time_4 = time.time()
	print ('predicting cost ',time_4 - time_3,' second','\n')



	print(test_predict)
	print(test_labels)
	print(test_predict.shape)
	print(test_labels.shape)

	score = accuracy_score(test_labels,test_predict)
	print ("The accruacy socre is ", score)
