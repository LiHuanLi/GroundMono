# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import torch
torch.cuda.init()
from torchvision import transforms, datasets

import networks
from PIL import Image
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

img_list = glob.glob('D:/data/kitti_data/2011_09_29/2011_09_29_drive_0004_sync/image_02/data/*.png')
#img_list = glob.glob('/mnt/data/dataset/kitti/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/*.png')
#'/mnt/data/dataset/kitti/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/*.png'
output_directory = 'results/original'
height = 192
width = 640

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_path = 'ablation_models/mono_640x192/'#
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_path = os.path.join(model_path, "depth.pth")

encoder_dict = torch.load(encoder_path)
encoder = networks.ResnetEncoder(18, False)
depth_net = networks.DepthDecoder(encoder.num_ch_enc,[0,1,2,3])

model_dict = encoder.state_dict()
encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
depth_net.load_state_dict(torch.load(depth_path))

encoder.cuda()
encoder.eval()
depth_net.cuda()
depth_net.eval()

# PREDICTING ON EACH IMAGE IN TURN
with torch.no_grad():
    for idx, image_path in enumerate(img_list):

        '''if image_path.split('/')[-1] != '0000000303.png':
            continue'''
        #print(image_path)
        # Load image and preprocess
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((width, height), pil.LANCZOS)
        input_image = transforms.ToTensor()(input_image).unsqueeze(0)

        # PREDICTION
        input_image = input_image.to(device)
        features = encoder(input_image)

        path = image_path.replace('kitti_data', 'segmentation').replace('data\\', '')

        obj_masks = torch.tensor(np.array(Image.open(path).resize((640, 192)))).unsqueeze(0).cuda()
        obj_masks = obj_masks >=24# 24
        #outputs = depth_net(features, obj_masks.float(), True)
        outputs = depth_net(features)

        disp = outputs['disp', 0]
        #gcp = disp * height * 1.92
        '''disp_resized = torch.nn.functional.interpolate(
            disp.unsqueeze(1), (original_height, original_width), mode="bilinear", align_corners=False)'''

        disp_resized_np = disp.squeeze().cpu().numpy()#disp.squeeze().cpu().numpy()#

        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')#magma
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(output_directory, os.path.basename(image_path))
        im.save(name_dest_im)

        #print("   Processed {:d} of {:d} images - saved predictions to:".format(
            #idx + 1, len(paths)))
        print("   - {}".format(name_dest_im))
        #print("   - {}".format(name_dest_npy))

print('-> Done!')

