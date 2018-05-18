import cv2
import glob
import numpy as np


def crop_img(f):
    oimg = cv2.imread(f)
    #print(oimg.shape, oimg.dtype)
    img_lab = cv2.cvtColor(oimg, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(20, 20))
    img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
    oimg = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # nimg = exposure.equalize_adapthist(oimg, clip_limit=0.03).astype(np.uint8)
    # cv2.imshow('asd', nimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #print(nimg.shape, nimg.dtype)
    # conver to YUV
    img_yuv = cv2.cvtColor(oimg, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    # convert back to BGR
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(src=img, ksize=(5, 5), sigmaX=0)

    _, thr = cv2.threshold(img, 150, 225, cv2.THRESH_BINARY)

    _, cnt, _ = cv2.findContours(image=thr, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    dst = np.zeros(img.shape)
    # cv2.drawContours(image=dst, contours=cnt, contourIdx=-1, color=[255, 255, 255], thickness=1)
    # cv2.imshow('contour on image', dst)

    max_area = float('-inf')
    longest_contour = None
    for c in cnt:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            longest_contour = c
    #print(longest_contour)
    # cv2.drawContours(image=oimg, contours=longest_contour, contourIdx=-1, color=[0, 0, 255], thickness=2)
    # cv2.imshow('contour on image', oimg)
    contour_perimeter = cv2.arcLength(curve=longest_contour, closed=True)
    approx = cv2.approxPolyDP(curve=longest_contour, epsilon=0.1*contour_perimeter, closed=True)

    bound_rect = cv2.boundingRect(longest_contour)
    #print(bound_rect)
    #cv2.rectangle(oimg, (bound_rect[0], bound_rect[1]), (bound_rect[0]+bound_rect[2], bound_rect[1]+bound_rect[3]), color=[0,255,0])
    #cv2.imshow('contour on image', oimg)
    x,y,w,h = bound_rect
    cropped = oimg[y:y+h, x:x+w]
    cv2.imwrite(f'cropped_{f}', cropped)


for f in glob.iglob('*.jpg'):
    crop_img(f)
