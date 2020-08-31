"""verify_txts.py

For verifying correctness of the generated YOLO txt annotations.
"""


import random
from pathlib import Path
from argparse import ArgumentParser

import cv2


WINDOW_NAME = "verify_txts"

parser = ArgumentParser()
parser.add_argument('dim', help='input width and height, e.g. 608x608')
args = parser.parse_args()

if random.random() < 0.5:
    print('Verifying test.txt')
    jpgs_path = Path('crowdhuman-%s/test.txt' % args.dim)
else:
    print('Verifying train.txt')
    jpgs_path = Path('crowdhuman-%s/train.txt' % args.dim)

with open(jpgs_path.as_posix(), 'r') as f:
    jpg_names = [l.strip()[5:] for l in f.readlines()]

random.shuffle(jpg_names)
for jpg_name in jpg_names:
    img = cv2.imread(jpg_name)
    img_h, img_w, _ = img.shape
    txt_name = jpg_name.replace('.jpg', '.txt')
    with open(txt_name, 'r') as f:
        obj_lines = [l.strip() for l in f.readlines()]
    for obj_line in obj_lines:
        cls, cx, cy, nw, nh = [float(item) for item in obj_line.split(' ')]
        color = (0, 0, 255) if cls == 0.0 else (0, 255, 0)
        x_min = int((cx - (nw / 2.0)) * img_w)
        y_min = int((cy - (nh / 2.0)) * img_h)
        x_max = int((cx + (nw / 2.0)) * img_w)
        y_max = int((cy + (nh / 2.0)) * img_h)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.imshow(WINDOW_NAME, img)
    if cv2.waitKey(0) == 27:
        break

cv2.destroyAllWindows()
