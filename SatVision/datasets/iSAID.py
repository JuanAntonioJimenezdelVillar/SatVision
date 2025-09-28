import os
import os.path as osp
import numpy as np
from PIL import Image
import cv2
from torch.utils import data
import torch
import datasets.edge_utils as edge_utils

import logging
from config import cfg

from utils.gsd_data import extract_gsd, extract_gsd_and_convert, tif_to_png

num_classes = 16
class_names = ['background', 'ship', 'store_tank', 'baseball_diamond', 'tennis_court', 'basketball_court',
               'Ground_Track_Field', 'Bridge', 'Large_Vehicle', 'Small_Vehicle', 'Helicopter',
               'Swimming_pool', 'Roundabout', 'Soccer_ball_field', 'plane', 'Harbor']
ignore_label = 255
root = cfg.DATASET.iSAID_DIR

label2trainid = {0: 255, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
                 11: 10, 12: 11, 13: 12, 14: 13, 15: 14}
id2cat = {0: 'background', 1: 'ship', 2: 'store_tank', 3: 'baseball_diamond', 4: 'tennis_court', 5: 'basketball_court',
          6: 'Ground_Track_Field', 7: 'Bridge', 8: 'Large_Vehicle', 9: 'Small_Vehicle', 10: 'Helicopter',
          11: 'Swimming_pool', 12: 'Roundabout', 13: 'Soccer_ball_field', 14: 'plane', 15: 'Harbor'}

palette = [0, 0, 0, 0, 0, 63, 0, 63, 63, 0, 63, 0, 0, 63, 127, 0, 63, 191, 0, 63, 255, 0, 127, 63, 0, 127, 127,
           0, 0, 127, 0, 0, 191, 0, 0, 255, 0, 191, 127, 0, 127, 191, 0, 127, 255, 0, 100, 155]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    new_mask = Image.fromarray(mask.astype(np.int8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

def make_dataset(quality, mode, hardnm=0):
    all_tokens = []
    img_paths = []
    gsd = None
    assert quality == 'semantic'
    assert mode in ['train', 'val', 'test', 'val_ori', 'val1000']

    image_path_base = osp.join(root, mode, 'images')
    mask_path_base = osp.join(root, mode, 'gray_masks') # O la ruta correcta de tus máscaras

    # --- Modificación para encontrar TIF o PNG ---
    image_files = os.listdir(image_path_base)
    image_files.sort()

    processed_basenames = set() # Para evitar duplicados si existen .tif y .png

    for img_filename in image_files:
        basename, ext = osp.splitext(img_filename)
        if basename in processed_basenames:
            continue # Ya procesamos este nombre base

        # Construye rutas potenciales
        potential_tif_path = osp.join(image_path_base, basename + '.tif')
        print(f"Potential TIF path: {potential_tif_path}")
        potential_png_path = osp.join(image_path_base, basename + '.png')

        # Determina la ruta de imagen principal a usar
        if osp.exists(potential_tif_path):
            #img_path = potential_tif_path # Prioriza TIF si existe
            img_path, gsd = extract_gsd_and_convert(potential_tif_path)
        elif osp.exists(potential_png_path):
            img_path = potential_png_path
        else:
            print(f"Warning: No image found for basename {basename}")
            continue # Saltar si no se encuentra ni TIF ni PNG

        # Construye la ruta de la máscara (ajusta el nombre según tu estructura)
        # Ejemplo: P0001_instance_color_RGB.png
        mask_filename = f"{basename}_instance_color_RGB.png"
        mask_path = osp.join(mask_path_base, mask_filename)

        token = (img_path, mask_path, gsd)
        print(f"Processing token: {token}")
        all_tokens.append(token)
        processed_basenames.add(basename)

    logging.info(f'iSAID has a total of {len(all_tokens)} images in {mode} phase')
    return all_tokens


class ISAIDDataset(data.Dataset):

    def __init__(self, quality, mode, maxSkip=0, joint_transform_list=None,
                 transform=None, target_transform=None, dump_images=False,
                 class_uniform_pct=None, class_uniform_title=0, test=False,
                 cv_split=None, scf=None, hardnm=0, edge_map=False, thicky=8):

        self.quality = quality
        self.mode = mode
        self.maxSkip = maxSkip
        self.joint_transform_list = joint_transform_list
        self.transform = transform
        self.target_transform = target_transform
        self.dump_images = dump_images
        self.class_uniform_pct = class_uniform_pct
        self.class_uniform_title = class_uniform_title
        if cv_split:
            self.cv_split = cv_split
            assert cv_split < cfg.DATASET.CV_SPLITS
        self.scf = scf
        self.hardnm = hardnm
        self.edge_map = edge_map
        self.data_tokens = make_dataset(quality, mode, hardnm)
        self.thicky = thicky

        assert len(self.data_tokens), 'Found 0 images please check the dataset'

    def __getitem__(self, index):

        token = self.data_tokens[index]
        image_path, mask_path, gsd = token
        image_name = osp.splitext(osp.basename(image_path))[0]
        

        if image_path.lower().endswith('.png'):
            image = Image.open(image_path).convert('RGB')

        else:
            print(f"Warning: Unrecognized image format for {image_path}")
            image = None
            
        if image is None:
            print(f"Could not load image for {image_name}. Returning None.")
            return None

        if self.mode != 'test':
            mask = Image.open(mask_path)
        else:
            mask = None
            
        if self.joint_transform_list is not None and mask is not None:
            for idx, xform in enumerate(self.joint_transform_list):
                image, mask = xform(image, mask)

            # Debug
        if self.dump_images:
            outdir = '../../dump_imgs_{}'.format(self.mode)
            os.makedirs(outdir, exist_ok=True)
            out_img_fn = os.path.join(outdir, image_name + '.png')
            out_msk_fn = os.path.join(outdir, image_name + '_mask.png')
            mask_img = colorize_mask(np.array(mask)) if mask is not None else None
            image.save(out_img_fn)
            if mask_img is not None:
                mask_img.save(out_msk_fn)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None and mask is not None:
            mask = self.target_transform(mask)


        if self.edge_map:
            boundary = self.get_boundary(mask, thicky=self.thicky)
            body = self.get_body(mask, boundary)
            return image, mask, body, boundary, image_name


        return image, mask, gsd, image_name

    def __len__(self):
        return len(self.data_tokens)

    def build_epoch(self):
        pass

    @staticmethod
    def get_boundary(mask, thicky=8):
        tmp = mask.data.numpy().astype('uint8')
        contour, _ = cv2.findContours(tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundary = np.zeros_like(tmp)
        boundary = cv2.drawContours(boundary, contour, -1, 1, thicky)
        boundary = boundary.astype(np.float)
        return boundary

    @staticmethod
    def get_body(mask, edge):
        edge_valid = edge == 1
        body = mask.clone()
        body[edge_valid] = ignore_label
        return body
