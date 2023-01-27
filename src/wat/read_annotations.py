# coding=utf-8
# @Project  ：keypoint-annotation-tool 
# @FileName ：read_annotation.py
# @Author   ：SoberReflection
# @Revision : sober 
# @Date     ：2023/1/27 12:40
import argparse
import json
import os
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
from tqdm import tqdm
from wat.tools import points2density


def apply_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float32)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def parse_cmdline_params():
    """
    @brief Parse command line parameters to get input and output file names.
    @param[in] argv Array of command line arguments.
    @return input and output file names if they were specified.
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', default='/Volumes/SoberSSD/SSD_Download/counting/output_1', help='Path to the output '
    #                                                                                              'directory.')
    parser.add_argument('--dir', required=True, help='Path to the output directory.')
    args = parser.parse_args()
    return args


def main():
    # Reading command line parameters
    args = parse_cmdline_params()
    print(args)

    vis_path = osp.join(args.dir, 'vis')
    os.makedirs(vis_path, exist_ok=True)

    files = defaultdict(list)
    for root, _, filenames in os.walk(args.dir):
        for filename in filenames:
            if filename.startswith('.') or 'vis' in root or 'input'  in root:
                continue

            name = osp.splitext(filename)[0]
            files[name].append(osp.join(root, filename))

    print('Total, there has %d images' % len(files.keys()))

    for name, vs in tqdm(files.items()):
        if len(vs) != 3:
            print('Error, the data %s must has image, json and npy mask'%name)
        else:
            raw_img, points, npy_norm = None, None, None
            for v in vs:
                if 'npy' in v:
                    npy_array = np.load(v)
                    min, max = npy_array.min(), npy_array.max()
                    npy_norm = (npy_array - min) / (max - min + 1e-8)
                elif 'json' in v:
                    with open(v, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                        points = np.asarray(json_data['points'])

                else:
                    raw_img = cv2.imread(v)
                    # raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

            if raw_img is None or points is None or npy_norm is None:
                print('Error, the data %s has problem' % name)

            mask = points2density(points, max_scale=3.0, max_radius=15.0, image_size=raw_img.shape[:2])
            min, max = mask.min(), mask.max()
            mask_norm = (mask - min) / (max - min + 1e-8)

            npy_vis = apply_scoremap(raw_img, npy_norm)
            mask_vis = apply_scoremap(raw_img, mask_norm)

            img = cv2.hconcat([raw_img, npy_vis, mask_vis])
            cv2.imwrite(osp.join(vis_path, name + '.jpg'), img)


if __name__ == "__main__":
    main()