import cv2
import numpy as np
import math

arr = []
for i in range(18):
	img = cv2.imread(f'm_{i+1:05d}.png')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

	groudtruth = np.zeros((15, 30, 3), dtype=np.int8)

	for row in range(15):
		for col in range(30):
			roi = img[row*10:row*10+10, col*10:col*10+10]
			# Red, Blue, Board
			votes = [0, 0, 0]
			for x in range(10):
				for y in range(10):
					if roi[x, y, 0]<20:
						votes[0]+=1
					elif roi[x, y, 0]>140:
						votes[1]+=1
					else:
						votes[2]+=1
			
			majority = max(votes)
			idx = votes.index(majority)

			if idx == 0:
				groudtruth[row][col][0] = 1
			elif idx == 1:
				groudtruth[row][col][1] = 1
			else:
				groudtruth[row][col][2] = 1

	arr.append(groudtruth)

arr = np.array(arr)
arr = np.reshape(arr, (-1, 1350))
np.savetxt('18.rb.300x150.label.txt', arr, fmt='%i', delimiter=',')
