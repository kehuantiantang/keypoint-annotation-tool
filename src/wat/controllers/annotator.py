#!/usr/bin/python
#
# @brief  This module and class represent a singleton controller that is
#         intended to store the temporary information about the current
#         annotation and save it to disk.
# @author Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   20 Jan 2021.

import ntpath
import os
import os.path as osp
import json
import cv2
import numpy as np
import threading

# My imports
import wat.common

from src.wat.tools import points2density


class BaseAnnotator(object):
    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name):
        return setattr(self.instance, name)


class TooltipAnnotator(BaseAnnotator):
    instance = None
    """
    @class that annotates surgical instrument tooltips. It only works for 
           scenes that contain a maximum of two tooltips. It is a Singleton.
    """
    class __TooltipAnnotator: 
        def __init__(self, data_dir, input_dir, output_dir, maxtips, gt_suffix):
            # Validate all the input information
            assert(data_dir is not None)
            assert(input_dir is not None)
            assert(output_dir is not None)
            assert(maxtips is not None)
            assert(gt_suffix is not None)
            
            # Store parameters in internal attributes
            self.data_dir = data_dir
            self.input_dir = os.path.join(self.data_dir, input_dir)
            self.output_dir = os.path.join(self.data_dir, output_dir)
            self.maxtips = maxtips
            self.gt_suffix = gt_suffix

            # Image-specific attributes
            self.path = None          # Path to the last file being annotated
            self.scale_factor = None  # Scale factor of the last file being annotated
            self.clicks = []          # [[x_0, y_0], [x_1, y_1], ... ]
            self.last_click_id = -1
            #self.clicks_mutex = threading.Lock()

        def new_image(self, path, scale_factor):
            self.path = path 
            self.scale_factor = scale_factor
            self.clicks = []
            self.last_click_id = -1


        def add_click(self, click_id, x, y, none_id='missing-tip'):
            #self.clicks_mutex.acquire()

            if click_id == none_id and len(self.clicks) < self.maxtips:
                self.clicks.append({'x': None, 'y': None})
            elif click_id != self.last_click_id and len(self.clicks) < self.maxtips:
                self.last_click_id = click_id
                #x_original = int(round(x / self.scale_factor))
                #y_original = int(round(y / self.scale_factor))
                self.clicks.append({'x': x, 'y': y})
            
            #self.clicks_mutex.release()

        def save(self):
            # Save JSON with the annotation
            im_fname = ntpath.basename(self.path)

            json_fname = wat.common.fname_no_ext(im_fname) + '.json'
            dst_path = os.path.join(self.output_dir, 'frames', json_fname)
            json_annotation = self._create_json_annotation()
            with open(dst_path, 'w', encoding='utf-8') as f:
                json.dump(json_annotation, f, indent=4)
            
            # Save binary mask with the tooltip annotation
            im_annot_fname = wat.common.fname_no_ext(im_fname) + '.npy'
            dst_path = os.path.join(self.output_dir, 'gt_density_map', im_annot_fname)
            im_annot = self._create_image_annotation()
            if im_annot is not None:
                np.save(dst_path, im_annot)
                img = cv2.imread(self.path)
                cv2.imwrite(os.path.join(self.output_dir, 'frames', wat.common.fname_no_ext(im_fname) + '.jpg'), img)

        
        def _get_original_clicks(self):
            original_clicks = []
            for click in self.clicks:
                if click['x'] is not None and click['y'] is not None:
                    original_clicks.append([
                        int(round(click['x'] / self.scale_factor)),
                        int(round(click['y'] / self.scale_factor)),
                    ])
                else:
                    original_clicks.append(click)
            return original_clicks

        def _create_json_annotation(self):
            filename = osp.split(self.path)[-1]
            return {'filename':filename, 'density': osp.splitext(filename)[0] + '.npy', 'points':
                                                                                     self._get_original_clicks()}


        def _create_image_annotation(self):
            # Get image dimensions
            if self.path is not None:
                im = cv2.imread(self.path)
                width = im.shape[1]
                height = im.shape[0]


                # Add tooltip annotations
                points = np.array(self._get_original_clicks())
                cnt_gt = points.shape[0]

                density = points2density(points, max_scale=3.0, max_radius=15.0, image_size=(height, width))
                if not cnt_gt == 0:
                    cnt_cur = density.sum()
                    density = density / cnt_cur * cnt_gt

                return density
            else:
                return None


    # Singleton implementation for the Controller class 
    def __init__(self, data_dir=None, input_dir='input', output_dir='output', 
                maxtips=2000, gt_suffix='_mask'):
        if TooltipAnnotator.instance is None:
            TooltipAnnotator.instance = TooltipAnnotator.__TooltipAnnotator(data_dir,
                input_dir, output_dir, maxtips, gt_suffix)
        else:
            # An annotator already exists, so you cannot pass different parameters 
            # than the ones you provided when it was built for first time, so just
            # in case, let's check that you are calling it properly 
            if data_dir is not None:
                assert(data_dir == TooltipAnnotator.instance.data_dir)

            assert(maxtips == TooltipAnnotator.instance.maxtips)
            assert(gt_suffix == TooltipAnnotator.instance.gt_suffix)

if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module cannot be run like a script.')
