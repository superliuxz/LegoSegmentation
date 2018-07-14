import cv2
import numpy as np
import math
from collections import defaultdict

def detect_edges(name):
    img = cv2.imread(name)
    img = cv2.resize(img, (400, 300))

    eq = False
    if eq:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    # cv2.imshow("Original", img)
    # cv2.waitKey()

    black = np.zeros(img.shape)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mid = np.median(gray)

    ksize = 23
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 5)

    cv2.imshow("Blurred", blurred)
    cv2.waitKey()

    edges = cv2.Canny(blurred, 10, 50, apertureSize=3)

    cv2.imshow("Edges", edges)
    cv2.waitKey()

    lines = cv2.HoughLines(image=edges, rho=1, theta=np.pi/180, threshold=62)
    # lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)

    print_lines = True
    if print_lines:
        black = np.zeros(img.shape)
        if lines is not None:
            for i in range(len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*a))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
                cv2.line(black, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
        
        cv2.imshow("Lines", black)
        cv2.waitKey()
    # print(lines)
    return lines, img

def segment_lines(lines):
    """
    https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    """
    angles = np.array([line[0][1] for line in lines])
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)
    labels, *_ = cv2.kmeans(data=np.array(pts),
                            K=2,
                            bestLabels=None,
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                            attempts=10,
                            flags=cv2.KMEANS_RANDOM_CENTERS)[1:]
    labels = labels.reshape(-1)
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented

def intersection(line1, line2):
    """
    Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

def segmented_intersections(lines):
    """
    Finds the intersections between groups of lines.
    """
    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))
    return intersections

def merge_points(points):
    # print(points)
    merged = [points[0]]
    for i, p in enumerate(points):
        if i > 0:
            for m in merged:
                dis = ((p[0]-m[0])**2+(p[1]-m[1])**2)**0.5
                if dis < 10:
                    m[0], m[1] = (m[0]+p[0])//2, (m[1]+p[1])//2
                    break
            else:
                merged.append(p)
    return merged

def sort_points(points: list):
    """
    sort the points into top left - btm left - btm right - top right
    order, such that the polygon is a rectangle
    """
    top_left = sorted(points, key=lambda x: (x[0]**2+x[1]**2)**0.5)[0]
    points.remove(top_left)
    btm_right = sorted(points, key=lambda x: (x[0]**2+x[1]**2)**0.5, reverse=True)[0]
    points.remove(btm_right)
    btm_left = sorted(points, key=lambda x: x[0], reverse=True)[0]
    points.remove(btm_left)
    top_right = points[0]
    return np.array([top_left, btm_left, btm_right, top_right], dtype=np.int32)

if __name__ == '__main__':
    for i in range(18, 19):
        print(f'working on {i+1:05d}.jpg')
        lines, img = detect_edges(f'{i+1:05d}.jpg')
        segmented = segment_lines(lines)

        intersections = segmented_intersections(segmented)

        ROI = merge_points(intersections)

        # sorted_ROI = sort_points(ROI)
        # print(sorted_ROI)
        up = min(ROI, key=lambda x: x[0])[0]
        down = max(ROI, key=lambda x: x[0])[0]
        left = min(ROI, key=lambda x: x[1])[1]
        right = max(ROI, key=lambda x: x[1])[1]
        # print(up, down, left, right)
        cropped = img[left:right, up:down]

        cv2.imshow("Cropped", cropped)
        cv2.waitKey()

        # msk = np.zeros(img.shape).astype(np.uint8)
        # result = cv2.fillPoly(msk, [sorted_ROI], (255,255,255))
        # masked_image = cv2.bitwise_and(img, msk)
        cropped = cv2.resize(cropped, (30, 15))

        cv2.imwrite(f'm_{i+1:05d}.png',cropped)
