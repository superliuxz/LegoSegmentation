import cv2

for i in range(5000):
	print(f'working on {i+1:04d}.png')
	img = cv2.imread(f'{i+1:04d}.png')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	blue = img.copy()
	yellow = img.copy()

	# royalblue
	blue[(img==[65, 105, 225]).all(axis=-1)] = 0
	blue[(img!=[65, 105, 225]).all(axis=-1)] = 255
	# golden
	yellow[(img==[255, 215, 0]).all(axis=-1)] = 0
	yellow[(img!=[255, 215, 0]).all(axis=-1)] = 255

	blue = cv2.cvtColor(blue, cv2.COLOR_RGB2GRAY)
	yellow = cv2.cvtColor(yellow, cv2.COLOR_RGB2GRAY)

	cv2.imwrite(f'{i+1:04d}b.png', blue)
	cv2.imwrite(f'{i+1:04d}y.png', yellow)
print('done')