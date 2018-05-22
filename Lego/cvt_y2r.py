import cv2


im = cv2.imread('test3.jpg')
im = cv2.resize(im, (400, 300))
print(im.shape)
lower = np.array([180, 150, 0])
upper = np.array([250, 250, 50])

cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
