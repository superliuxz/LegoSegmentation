import cv2
import numpy as np


def transform(pos):
    # This function is used to find the corners of the object and the dimensions of the object
    pts = []
    n = len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))

    sums = {}
    diffs = {}

    for i in pts:
        x = i[0]
        y = i[1]
        sum = x + y
        diff = y - x
        sums[sum] = i
        diffs[diff] = i
    sums = sorted(sums.items())
    diffs = sorted(diffs.items())
    n = len(sums)
    rect = [sums[0][1], diffs[0][1], diffs[n - 1][1], sums[n - 1][1]]
    #	   top-left   top-right   bottom-left   bottom-right

    h1 = np.sqrt((rect[0][0] - rect[2][0]) ** 2 + (rect[0][1] - rect[2][1]) ** 2)  # height of left side
    h2 = np.sqrt((rect[1][0] - rect[3][0]) ** 2 + (rect[1][1] - rect[3][1]) ** 2)  # height of right side
    h = max(h1, h2)

    w1 = np.sqrt((rect[0][0] - rect[1][0]) ** 2 + (rect[0][1] - rect[1][1]) ** 2)  # width of upper side
    w2 = np.sqrt((rect[2][0] - rect[3][0]) ** 2 + (rect[2][1] - rect[3][1]) ** 2)  # width of lower side
    w = max(w1, w2)

    return int(w), int(h), rect


img = cv2.imread('example.jpg')
img = cv2.resize(src=img, dsize=(500, 500))

cv2.imshow('original', img)
# convert to gray scale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
# blur
gray = cv2.GaussianBlur(src=gray, ksize=(11, 11), sigmaX=0)
# use Canny to detect edge; returns a 2(3)d ndarray, same size as the image
edge = cv2.Canny(image=gray, threshold1=100, threshold2=200)
cv2.imshow('edge', edge)


# find contour from edge
# the result of Canny edge detection is an 'area', therefore if pass the Canny edge into findContours, it will return
# two contours, one being the inner contour and one being the outer contour
_, contours, _ = cv2.findContours(image=edge, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image=img, contours=contours, contourIdx=-1, color=[0, 255, 0], thickness=2)
cv2.imshow('contour on image', img)


# `contours` has two set of contours
# one outer one inner
# find the outer one (which corresponds to the max area out of the two)
max_area = float('-inf')
longest_contour = None
for c in contours:
    area = cv2.contourArea(c)
    if area > max_area:
        max_area = area
        longest_contour = c

# perimeter aka arcLength
contour_perimeter = cv2.arcLength(curve=longest_contour, closed=True)
# connect the dots in longest_contour; becoz closed=True, it returns four corners of the rectangle
approx = cv2.approxPolyDP(curve=longest_contour, epsilon=0.02*contour_perimeter, closed=True)

# size = img.shape
# w, h, arr = transform(approx)
#
# pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
# pts1 = np.float32(arr)
# M = cv2.getPerspectiveTransform(pts1, pts2)
# dst = cv2.warpPerspective(img, M, (w, h))
# image = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
# image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
# cv2.imshow('cropped', image)
cv2.waitKey(0)
cv2.destroyAllWindows()