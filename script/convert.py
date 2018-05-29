from os import listdir, makedirs
from os.path import isfile, join, exists

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="Data input directory")
parser.add_argument("output_dir", help="Data output directory")
args = parser.parse_args()

import numpy as np
from scipy import misc
from PIL import Image

scale = 2.0
patch_size = 200
stride = 200

if not exists(args.output_dir):
    makedirs(args.output_dir)
if not exists(join(args.output_dir, "input")):
    makedirs(join(args.output_dir, "input"))
if not exists(join(args.output_dir, "label")):
    makedirs(join(args.output_dir, "label"))

count = 1

def crop(scaled, image, y, x):
    print(y,x)
    sub_img = scaled[y : y + patch_size, x : x + patch_size]
    sub_img_label = image[y : y + patch_size, x : x + patch_size]
    misc.imsave(join(args.output_dir, "input", str(count) + '.png'), sub_img)
    misc.imsave(join(args.output_dir, "label", str(count) + '.png'), sub_img_label)

for f in listdir(args.input_dir):
    f = join(args.input_dir, f)
    if not isfile(f):
        continue

    image = np.asarray(Image.open(f).convert('RGB'))
    h, w, c = image.shape
    print('h:w = %d:%d, patch:%d, stride:%d' % (h,w,patch_size,stride))

    scaled = misc.imresize(image, 1.0/scale, 'bicubic')
    scaled = misc.imresize(scaled, scale/1.0, 'bicubic')

    num_x = int((w - patch_size)/stride) + 1
    num_y = int((h - patch_size)/stride) + 1
    rem_w = (w - patch_size) % stride
    rem_h = (h - patch_size) % stride
    print(num_x, num_y, rem_w, rem_h)

    for y in range(0, num_y * patch_size, stride):
        for x in range(0, num_x * patch_size, stride):
            crop(scaled, image, y, x)
            count += 1

    if rem_w > 0:
        for y in range(0, num_y * patch_size, stride):
            x = w - patch_size
            crop(scaled, image, y, x)
            count += 1

    if rem_h > 0:
        y = h - patch_size
        for x in range(0, num_x * patch_size, stride):
            crop(scaled, image, y, x)
            count += 1

    if rem_w > 0 and rem_h > 0:
        y = h - patch_size
        x = w - patch_size
        crop(scaled, image, y, x)
        count += 1

