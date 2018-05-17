import argparse
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument('ftype', type=str, help='the image file extensions, such as "*.jpg"')
parser.add_argument('offset', type=int, help='the files are renamed after this integer')
args = parser.parse_args()

counter = args.offset
ftype = args.ftype

for f in glob.iglob(ftype):
    fname, ext = os.path.splitext(os.path.basename(f))
    print(f'rename {fname} to {counter:05d}')
    os.rename(f, f'{counter:05d}{ext}')
    counter += 1