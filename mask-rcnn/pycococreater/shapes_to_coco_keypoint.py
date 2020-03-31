#!/usr/bin/env python3

import datetime
import json
import os
import sys
import re
import fnmatch
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
parser.add_argument('--option', required=True,
                    default="option.json",
                    metavar="<option>",
                    help='option json parser (default=option.json)')
parser.add_argument('--path', required=True,
                    metavar="<string>",
                    help='Original Iamge Path')
parser.add_argument('--root_path', required=True,
                    metavar="<string>",
                    help='Root Path')
parser.add_argument('--keypoints', required=False,
                    default="",
                    metavar="<string>",
                    help='keypoint')

args = parser.parse_args()

ROOT_DIR = args.root_path;
IMAGE_DIR = os.path.join(ROOT_DIR, args.path)
ANNOTATION_DIR = os.path.join(ROOT_DIR, args.path+'_labels')
if args.keypoints is not "" :
    KEYPOINTS_LIST = np.load(ROOT_DIR + "/" + args.path+"_"+args.keypoints).tolist()

print(ROOT_DIR + "/" + args.path+"_"+args.option)
data = json.loads( open(ROOT_DIR + "/" + args.path+"_"+args.option).read() )

INFO = data['INFO']
LICENSES = data['LICENSES']
CATEGORIES = data['CATEGORIES']

def filter_for_jpeg(root, files):
    file_types = ['*.JPG', '*.jpg','*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    
    return files

def filter_for_annotations(root, files, image_filename):
    file_types = ['*.PNG', '*.png', '*.JPG', '*.jpg']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def main():

    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    
    # filter for jpeg images
    files = sorted( os.listdir(IMAGE_DIR) )
    image_files = filter_for_jpeg(IMAGE_DIR, files)

    # go through each image
    for image_filename in image_files:
        image = Image.open(image_filename)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)
        coco_output["images"].append(image_info)

        # filter for associated png annotations
        files = sorted( os.listdir(ANNOTATION_DIR) )
        annotation_files = filter_for_annotations(ANNOTATION_DIR, files, image_filename)

        # go through each associated annotation
        for annotation_filename in annotation_files:
            
            print(annotation_filename)
            class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

            category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
            binary_mask = np.asarray(Image.open(annotation_filename)
                .convert('1')).astype(np.uint8)

            annotation_info = pycococreatortools.create_annotation_info(
                segmentation_id, image_id, category_info, binary_mask,
                image.size, tolerance=2, data_type=args.path)

            if args.keypoints is not "" :
                annotation_info["keypoints"] = KEYPOINTS_LIST[segmentation_id-1]
                annotation_info["num_keypoints"] = "17"

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)


            segmentation_id = segmentation_id + 1

        image_id = image_id + 1

    if not os.path.isdir("{}/annotations".format(ROOT_DIR)):
        os.makedirs("{}/annotations".format(ROOT_DIR))

    with open('{}/annotations/instances_{}.json'.format(ROOT_DIR, args.path), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    main()
