"""gen_txts.py

To generate YOLO txt files from the original CrowdHuman annotations.
Please also refer to README.md in this directory.

Inputs:
    * raw/annotation_train.odgt
    * raw/annotation_val.odgt
    * crowdhuman-512x512/[IDs].jpg

Outputs:
    * crowdhuman-512x512/train.txt
    * crowdhuman-512x512/test.txt
    * crowdhuman-512x512/[IDs].txt (one annotation for each image in train or test)
"""


import json
from pathlib import Path

import numpy as np
import cv2


# These are the expected input image width/height of the yolov4 model
INPUT_WIDTH  = 512
INPUT_HEIGHT = 512

# Input/output directories
IMAGES_DIR = 'crowdhuman-%dx%d' % (INPUT_WIDTH, INPUT_HEIGHT)
OUTPUT_DIR = 'crowdhuman-%dx%d' % (INPUT_WIDTH, INPUT_HEIGHT)

# Minimum width/height of objects for detection (don't learn from
# objects smaller than these
MIN_W = 6
MIN_H = 6

# Do K-Means clustering in order to determine "anchor" sizes
DO_KMEANS = True
KMEANS_CLUSTERS = 9
BBOX_WHS = []  # keep track of bbox width/height with respect to 608x608


def image_shape(ID):
    images_dir = Path(IMAGES_DIR)
    jpg_path = images_dir / ('%s.jpg' % ID)
    img = cv2.imread(jpg_path.as_posix())
    return img.shape


def txt_line(cls, bbox, img_w, img_h):
    """Generate 1 line in the txt file."""
    x, y, w, h = bbox
    x = max(int(x), 0)
    y = max(int(y), 0)
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)
    if w < MIN_W or h < MIN_H:
        return ''
    else:
        if DO_KMEANS:
            global BBOX_WHS
            BBOX_WHS.append(
                (
                    float(w) * INPUT_WIDTH  / img_w,
                    float(h) * INPUT_HEIGHT / img_h
                )
            )
        cx = (x + w / 2.) / img_w
        cy = (y + h / 2.) / img_h
        nw = float(w) / img_w
        nh = float(h) / img_h
        return '%d %.6f %.6f %.6f %.6f\n' % (cls, cx, cy, nw, nh)


def process(set_='test', annotation_filename='raw/annotation_val.odgt'):
    """Process either 'train' or 'test' set."""
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    jpgs = []
    with open(annotation_filename, 'r') as fanno:
        for raw_anno in fanno.readlines():
            anno = json.loads(raw_anno)
            ID = anno['ID']  # e.g. '273271,c9db000d5146c15'
            print('Processing ID: %s' % ID)
            img_h, img_w, img_c = image_shape(ID)
            assert img_c == 3  # should be a BGR image
            txt_path = output_dir / ('%s.txt' % ID)
            # write a txt for each image
            with open(txt_path.as_posix(), 'w') as ftxt:
                for obj in anno['gtboxes']:
                    if obj['tag'] == 'mask':
                        continue  # ignore non-human
                    assert obj['tag'] == 'person'
                    if 'hbox' in obj.keys():  # head
                        line = txt_line(0, obj['hbox'], img_w, img_h)
                        if line:
                            ftxt.write(line)
                    if 'fbox' in obj.keys():  # full body
                        line = txt_line(1, obj['fbox'], img_w, img_h)
                        if line:
                            ftxt.write(line)
            jpgs.append('data/%s/%s.jpg' % (output_dir, ID))
    # write the 'data/crowdhuman/train.txt' or 'data/crowdhuman/test.txt'
    set_path = output_dir / ('%s.txt' % set_)
    with open(set_path.as_posix(), 'w') as fset:
        for jpg in jpgs:
            fset.write('%s\n' % jpg)


def rm_tree(path_name):
    """Remove a path recursively."""
    pth = Path(path_name)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


def rm_txts(path_name):
    """Remove txt files in path."""
    pth = Path(path_name)
    for txt in pth.glob('*.txt'):
        if txt.is_file():
            txt.unlink()


def main():
    images_dir = Path(IMAGES_DIR)
    if not images_dir.is_dir():
        raise SystemExit('ERROR: %s does not exist.' % IMAGES_DIR)
    rm_txts(OUTPUT_DIR)

    process('test', 'raw/annotation_val.odgt')
    process('train', 'raw/annotation_train.odgt')

    if DO_KMEANS:
        try:
            from sklearn.cluster import KMeans
        except ModuleNotFoundError:
            print('WARNING: no sklearn, skipping anchor clustering...')
        else:
            X = np.array(BBOX_WHS)
            kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=0).fit(X)
            centers = kmeans.cluster_centers_
            centers = centers[centers[:, 0].argsort()]  # sort by bbox w
            print('\n** for yolov4-%dx%d, ' % (INPUT_WIDTH, INPUT_HEIGHT), end='')
            print('resized bbox width/height clusters are: ', end='')
            print(' '.join(['(%.2f, %.2f)' % (c[0], c[1]) for c in centers]))
            print('\nanchors = ', end='')
            print(',  '.join(['%d,%d' % (int(c[0]), int(c[1])) for c in centers]))


if __name__ == '__main__':
    main()
