
#'backend' functions necessary for reading data, images and reconstruction 

from brian2 import *
from time import time
import parameters as par
import numpy as np
import pandas as pd
import cv2

# made this function to read images to be trained (using csv file in this example instead)
def read_img():

	img = cv2.imread("data/101.png", 0)
	img2 = cv2.imread("data/101.png",0)

	# img = cv2.imread("data/camera-icon.png", 0)
	# img2 = cv2.imread("data/camera-icon.png",0)

	img = np.ndarray.flatten(img)
	img2 = np.ndarray.flatten(img2)
	# data = []
	# data.append(pd.DataFrame(img))
	# data.append(pd.DataFrame(img2))
	# data.append(pd.DataFrame(img2))
	# data.to_csv('train_test.csv')

	# print(img)
	# numpy.savetxt("train1.csv", img, delimiter=",")

	# df = pd.DataFrame(img)
	# df.to_csv('file.csv',index=False)

	return img

def recon_weights(W):

	for i in range(par.hid_size):
		temp = W[i]
		recon = np.reshape(temp,(par.x_pixel,par.x_pixel))*255
		cv2.imwrite("weights/weightscamfinal_sick" + str(i) + ".png",recon)

def read_dataset():
	df = pd.read_csv("data/train.csv")
	
	# read_img()
	# df = pd.read_csv('file.csv')
	
	df = df.as_matrix()
	X = df[:,1:]
	y = df[:,0]
	return X,y