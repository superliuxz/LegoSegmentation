import argparse
import cv2
import glob

parser = argparse.ArgumentParser()
parser.add_argument('w', type=int, help='new width')
parser.add_argument('h', type=int, help='new height')
args = parser.parse_args()

for f in glob.iglob('*.jpg'):
    img = cv2.imread(f)
    img = cv2.resize(src=img, dsize=(args.w, args.h), interpolation=cv2.INTER_AREA)
    cv2.imwrite('s_'+f, img)
