"""
Evaluation Script
Support Two Modes: Pooling based inference and sliding based inference
Pooling based inference is simply whole image inference.
"""
import os
import logging
import sys
import argparse
import re
import queue
import threading
from math import ceil
from datetime import datetime
from scipy import ndimage
from tqdm import tqdm
import cv2
from PIL import Image
import PIL

from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms

import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import transforms.transforms as extended_transforms

from config import assert_and_infer_cfg
from datasets import iSAID, Posdam, Vaihingen
from optimizer import restore_snapshot
import transforms.joint_transforms as joint_transforms

from utils.my_data_parallel import MyDataParallel
from utils.misc import fast_hist, save_log, evaluate_eval_for_inference, speed_test, per_class_iu

import network

sys.path.append(os.path.join(os.getcwd()))
sys.path.append(os.path.join(os.getcwd(), '../'))

parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--dump_images', action='store_true', default=False)
parser.add_argument('--arch', type=str, default='', required=True)
parser.add_argument('--single_scale', action='store_true', default=False)
parser.add_argument('--scales', type=str, default='0.5,1.0,2.0')
parser.add_argument('--dist_bn', action='store_true', default=False)
parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--fixed_aspp_pool', action='store_true', default=False,
                    help='fix the aspp image-level pooling size to 105')

parser.add_argument('--sliding_overlap', type=float, default=1 / 3)
parser.add_argument('--no_flip', action='store_true', default=False,
                    help='disable flipping')
parser.add_argument('--dataset', type=str, default='cityscapes',
                    help='cityscapes, video_folder')
parser.add_argument('--dataset_cls', type=str, default='cityscapes', help='cityscapes')
parser.add_argument('--trunk', type=str, default='resnet101', help='cnn trunk')
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='Dataset Location')
parser.add_argument('--split', type=str, default='val')
parser.add_argument('--crop_size', type=int, default=513)
parser.add_argument('--exp_name', type=str, default=None)
parser.add_argument('--snapshot', type=str, default='')
parser.add_argument('--ckpt_path', type=str, default=None)
parser.add_argument('-im', '--inference_mode', type=str, default='sliding',
                    help='sliding or pooling or whole')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (4 items evaluated) to verify nothing failed')
parser.add_argument('--cv_split', type=int, default=None)
parser.add_argument('--mode', type=str, default='fine')
parser.add_argument('--split_index', type=int, default=0)
parser.add_argument('--split_count', type=int, default=1)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--resume', action='store_true', default=False,
                    help='Resume Inference')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Only in pooling mode')
parser.add_argument('--maxpool_size', type=int, default=9)
parser.add_argument('--avgpool_size', type=int, default=9)
parser.add_argument('--edge_points', type=int, default=32)
parser.add_argument('--resize_scale', type=int)
parser.add_argument('--local_rank', default=0, type=int,
                    help='parameter used by distributed library')
parser.add_argument('--match_dim', default=64, type=int, help='dim when match in pfnet')
parser.add_argument('--ignore_background', action='store_true', help='whether to ignore background class when '
                                                                     'generating coarse mask in pfnet')
parser.add_argument('--with_f1', action='store_true', default=False)
parser.add_argument('--beta', default=1, type=int)
parser.add_argument('--speed_test', action='store_true')

args = parser.parse_args()
assert_and_infer_cfg(args, train_mode=False)
args.apex = False  # No support for apex eval
cudnn.benchmark = False
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
date_str = str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))

def sliding_window_cropping(data, scale=1.0):
    """
    Sliding Window Cropping
    Take the image and create a mapping and multiple crops
    """
    sliding_window_cropping = None
    mapping = {}
    crop_ctr = 0
    if scale < 1.0:
        scale = 1.0
    tile_size = (int(args.crop_size * scale), int(args.crop_size * scale))

    overlap = args.sliding_overlap

    for img_ctr in range(len(data)):

        h, w = data[img_ctr].shape[1:]
        mapping[img_ctr] = [w, h, []]
        if overlap <= 1:
            stride = ceil(tile_size[0] * (1 - overlap))
        else:
            stride = tile_size[0] - overlap

        tile_rows = int(
            ceil((w - tile_size[0]) / stride) + 1)
        tile_cols = int(ceil((h - tile_size[1]) / stride) + 1)
        for row in range(tile_rows):
            for col in range(tile_cols):
                y1 = int(col * stride)
                x1 = int(row * stride)
                x2 = min(x1 + tile_size[1], w)
                y2 = min(y1 + tile_size[0], h)
                x1 = int(x2 - tile_size[1])
                y1 = int(y2 - tile_size[0])
                if x1 < 0:  # for portrait the x1 underflows sometimes
                    x1 = 0
                if y1 < 0:
                    y1 = 0
                if crop_ctr == 0:
                    sliding_window_cropping = data[img_ctr][:, y1:y2, x1:x2].unsqueeze(0)

                else:
                    sliding_window_cropping = torch.cat(
                        (sliding_window_cropping,
                         data[img_ctr][:, y1:y2, x1:x2].unsqueeze(0)),
                        dim=0)

                mapping[img_ctr][2].append((x1, y1, x2, y2))
                crop_ctr += 1

    return (mapping, sliding_window_cropping)


def resize_thread(flip, index, array, resizequeue, origw, origh):
    """
    Thread to resize the image size
    """
    if flip:
        resizequeue.put((index, cv2.resize(np.fliplr(array),
                                           (origw, origh),
                                           interpolation=cv2.INTER_LINEAR)))
    else:
        resizequeue.put((index, cv2.resize(array, (origw, origh),
                                           interpolation=cv2.INTER_LINEAR)))


def reverse_mapping(i, ctr, input_img, mapping, que, flip, origw, origh):
    """
    Reverse Mapping for sliding window
    """
    w, h, coords = mapping[i]
    full_probs = np.zeros((args.dataset_cls.num_classes, h, w))
    count_predictions = np.zeros((args.dataset_cls.num_classes, h, w))
    for j in range(len(coords)):
        x1, y1, x2, y2 = coords[j]
        count_predictions[y1:y2, x1:x2] += 1
        average = input_img[ctr]
        if full_probs[:, y1: y2, x1: x2].shape != average.shape:
            average = average[:, :y2 - y1, :x2 - x1]

        full_probs[:, y1:y2, x1:x2] += average
        ctr = ctr + 1

    # Accumulate and average overerlapping areas
    full_probs = full_probs / count_predictions.astype(np.float)
    out_temp = []
    out_y = []
    t_list = []
    resizequeue = queue.Queue()
    classes = full_probs.shape[0]
    for y_ in range(classes):
        t = threading.Thread(target=resize_thread, args=(flip, y_, full_probs[y_],
                                                         resizequeue, origw, origh))
        t.daemon = True
        t.start()
        t_list.append(t)

    for thread in t_list:
        thread.join()
        out_temp.append(resizequeue.get())

    dictionary = dict(out_temp)
    for iterator in range(classes):
        out_y.append(dictionary[iterator])

    que.put(out_y)


def reverse_sliding_window(mapping, input_img, flip_list, origw, origh, final_queue):
    """
    Take mapping and crops and reconstruct original image
    """

    batch_return = []
    ctr = 0
    # Loop through the maps and merge them together
    que = queue.Queue()
    t_list = []
    for i in range(len(mapping)):
        t = threading.Thread(target=reverse_mapping, args=(i, ctr, input_img, mapping, que,
                                                           flip_list[i], origw, origh))
        ctr = ctr + len(mapping[i][2])
        t.daemon = True
        t.start()
        t_list.append(t)

    for item in t_list:
        item.join()
        batch_return.append(que.get())

    final_queue.put(np.mean(batch_return, axis=0))


def pooled_eval(model, image, scale):
    """
    Perform Pooled Evaluation
    """
    with torch.no_grad():
        y = model(image)
        if scale > 1.0:
            y = [torch.nn.AvgPool2d((2, 2), stride=2)(y_) for y_ in y]
        elif scale < 1.0:
            y = [torch.nn.Upsample(scale_factor=2, mode='bilinear')(y_) for y_ in y]
        else:
            pass

    return y


def flip_tensor(x, dim):
    """
    Flip Tensor along a dimension
    """
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]


def inference_pool(model, img, scales):
    """
    Post Inference Pool Operations
    """

    if args.no_flip:
        flip_range = 1
    else:
        flip_range = 2

    y_tmp_with_flip = 0
    for flip in range(flip_range):
        y_tmp = None
        for i in range(len(scales)):
            if type(y_tmp) == type(None):
                y_tmp = pooled_eval(model, img[flip][i], scales[i])
            else:
                out = pooled_eval(model, img[flip][i], scales[i])
                [x.add_(y) for x, y in zip(y_tmp, out)]
        if flip == 0:
            y_tmp_with_flip = y_tmp
        else:
            [x.add_(flip_tensor(y, 3)) for x, y in zip(y_tmp_with_flip, y_tmp)]

    y = [torch.argmax(y_ / (flip_range * len(scales)), dim=1).cpu().numpy() for y_ in
         y_tmp_with_flip]

    return y


def inference_sliding(model, img, scales):
    """
    Sliding Window Inference Function
    """

    w, h = img.size
    origw, origh = img.size
    preds = []
    if args.no_flip:
        flip_range = 1
    else:
        flip_range = 2

    finalque = queue.Queue()
    t_list = []
    for scale in scales:

        target_w, target_h = int(w * scale), int(h * scale)
        scaled_img = img.resize((target_w, target_h), Image.BILINEAR)
        image_list = []
        flip_list = []
        for flip in range(flip_range):
            if flip:
                scaled_img = scaled_img.transpose(Image.FLIP_LEFT_RIGHT)

            img_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*mean_std)])
            image = img_transform(scaled_img)
            image_list.append(image)
            flip_list.append(flip)

        mapping, input_crops = sliding_window_cropping(image_list, scale=scale)
        torch.cuda.empty_cache()
        with torch.no_grad():
            bi, _, hi, wi = input_crops.size()
            if hi >= args.crop_size:
                output_scattered_list = []
                for b_idx in range(bi):
                    cur_input = input_crops[b_idx,:,:,:].unsqueeze(0).cuda()
                    cur_output = model(cur_input)
                    output_scattered_list.append(cur_output)
                output_scattered = torch.cat(output_scattered_list, dim=0)
            else:
                input_crops = input_crops.cuda()
                output_scattered = model(input_crops)

        output_scattered = output_scattered.data.cpu().numpy()

        t = threading.Thread(target=reverse_sliding_window, args=(mapping, output_scattered,
                                                                  flip_list, origw,
                                                                  origh, finalque))
        t.daemon = True
        t.start()
        t_list.append(t)

    for threads in t_list:
        threads.join()
        preds.append(finalque.get())

    return preds


def inference_whole(model, img, scales):
    """
        whole images inference
    """
    w, h = img.size
    origw, origh = img.size
    preds = []
    if args.no_flip:
        flip_range = 1
    else:
        flip_range = 2

    for scale in scales:
        target_w, target_h = int(w * scale), int(h * scale)
        scaled_img = img.resize((target_w, target_h), Image.BILINEAR)

        for flip in range(flip_range):
            if flip:
                scaled_img = scaled_img.transpose(Image.FLIP_LEFT_RIGHT)

            img_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(*mean_std)])
            image = img_transform(scaled_img)
            with torch.no_grad():
                input = image.unsqueeze(0).cuda()
                scale_out = model(input)
                scale_out = F.upsample(scale_out, size=(origh, origw), mode="bilinear", align_corners=True)
                scale_out = scale_out.squeeze().cpu().numpy()
                if flip:
                    scale_out = scale_out[:, :, ::-1]
            preds.append(scale_out)

    return preds

def custom_collate_fn(batch):
    images, masks, gsds, image_names = zip(*batch)
    images = torch.stack(images, 0)
    masks = [mask for mask in masks if mask is not None]
    if masks:
        masks = torch.stack(masks, 0)
    else:
        masks = None
    return images, masks, gsds, image_names

def setup_loader():
    """
    Setup Data Loaders
    """
    val_input_transform = transforms.ToTensor()
    target_transform = extended_transforms.MaskToTensor()
    val_joint_transform_list = [joint_transforms.Resize(args.resize_scale)]
    if args.dataset == 'iSAID':
        args.dataset_cls = iSAID
        test_set = args.dataset_cls.ISAIDDataset(args.mode, args.split,
                                                 joint_transform_list=val_joint_transform_list,
                                                 transform=val_input_transform,
                                                 target_transform=target_transform)
    elif args.dataset == 'Posdam':
        args.dataset_cls = Posdam
        test_set = args.dataset_cls.POSDAMDataset(args.mode, args.split,
                                                 joint_transform_list=val_joint_transform_list,
                                                 transform=val_input_transform,
                                                 target_transform=target_transform)
    elif args.dataset == 'Vaihingen':
        args.dataset_cls = Vaihingen
        test_set = args.dataset_cls.VAIHINGENDataset(args.mode, args.split,
                                                     joint_transform_list=val_joint_transform_list,
                                                     transform=val_input_transform,
                                                     target_transform=target_transform)
    else:
        raise NameError('-------------Not Supported Currently-------------')

    if args.split_count > 1:
        test_set.split_dataset(args.split_index, args.split_count)

    batch_size = 1
    if args.inference_mode == 'pooling':
        batch_size = args.batch_size

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=args.num_workers,
                             shuffle=False, pin_memory=False, drop_last=False, collate_fn=custom_collate_fn)

    return test_loader


def get_net():
    """
    Get Network for evaluation
    """
    logging.info('Load model file: %s', args.snapshot)
    print(args)
    net = network.get_net(args, criterion=None)
    if args.inference_mode == 'pooling':
        net = MyDataParallel(net, gather=False).cuda()
    else:
        net = torch.nn.DataParallel(net).cuda()
    if args.speed_test:
        net.eval()
        speed_test(model=net, size=args.test_size)
        return
    net, _ = restore_snapshot(net, optimizer=None,
                              snapshot=args.snapshot, restore_optimizer_bool=False)
    net.eval()
    return net

class StatisticsTracker:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.class_counts = np.zeros(num_classes, dtype=np.int64)
        self.total_objects = 0
        self.total_image_area = 0

        self.density_areas = []
        self.object_physical_areas = {i: [] for i in range(self.num_classes)}
        self.object_dimensions = {i: [] for i in range(self.num_classes)}
        self.area_classifications = {
            'Muy Densa': 0,
            'Densa': 0,
            'Dispersa': 0
        }
        self.density_colors = {
            'Muy Densa': [1, 0, 0, 0.7], # Red
            'Densa': [1, 0.65, 0, 0.7],  # Orange
            'Dispersa': [0, 1, 0, 0.7] #Green
        }
        self.spatial_stats = {
            'arriba_izquierda': 0, 'arriba_centro': 0, 'arriba_derecha': 0,
            'medio_izquierda': 0, 'medio_centro': 0, 'medio_derecha': 0,
            'abajo_izquierda': 0, 'abajo_centro': 0, 'abajo_derecha': 0
        }
    
    def save_density_visualization(self, prediction, output_dir, img_name):
        height, width = prediction.shape
        total_mpx = (height * width) / 1e6
        
        # Debug total objects and area
        total_objects = 0
        for class_id in range(1, self.num_classes):
            class_mask = (prediction == class_id)
            if class_mask.any():
                _, num_objects = ndimage.label(class_mask)
                total_objects += num_objects
      
        total_density = total_objects / total_mpx
        
        cell_size = 100
        grid_h = height // cell_size
        grid_w = width // cell_size
        density_grid = np.zeros((grid_h, grid_w))
        
        total_cell_objects = 0
        
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * cell_size
                y_end = min((i + 1) * cell_size, height)
                x_start = j * cell_size
                x_end = min((j + 1) * cell_size, width)
                
                cell_area_mpx = ((y_end - y_start) * (x_end - x_start)) / 1e6
                cell_objects = 0
                
                for class_id in range(1, self.num_classes):
                    cell_mask = prediction[y_start:y_end, x_start:x_end] == class_id
                    if cell_mask.any():
                        _, num_objects = ndimage.label(cell_mask)
                        cell_objects += num_objects
                
                total_cell_objects += cell_objects
                density_grid[i, j] = cell_objects / cell_area_mpx if cell_area_mpx > 0 else 0
        
        print(f"Debug - Sum of cell objects: {total_cell_objects}")
        print(f"Debug - Verification - objects match: {total_objects == total_cell_objects}")
        
        max_density = min(total_density * 1.5, density_grid.max())
        
        plt.figure(figsize=(12, 8))
        plt.imshow(density_grid, cmap='RdYlGn_r', interpolation='bilinear', vmax=max_density, vmin=0)
        plt.colorbar(label='Objects per million pixels')
        plt.title(f'Object Density Heatmap\nTotal Objects: {total_objects}, Density: {total_density:.2f} obj/Mpx')
        
        density_viz_path = os.path.join(output_dir, f'{img_name}_density_heatmap.png')
        plt.savefig(density_viz_path, dpi=300, bbox_inches='tight')
        plt.close()

    def update(self, prediction, img_name, output_dir, gsd=None):
        height, width = prediction.shape
        MIN_PIXEL_COUNT = max(10, int(0.00002 * height * width))
        if gsd is not None:
            img_area = height * width * (gsd ** 2)
            self.total_image_area += img_area
        else:
            img_area = prediction.shape[0] * prediction.shape[1]

        objects_this_image = 0
        h_third = height // 3
        w_third = width // 3

        for class_id in range(1, self.num_classes):
            class_mask = (prediction == class_id)
            if class_mask.any():
                distance = ndimage.distance_transform_edt(class_mask)
                labeled_array, num_objects = ndimage.label(class_mask)
                valid_object_count = 0
                for obj_id in range(1, num_objects + 1):
                    obj_mask = labeled_array == obj_id
                    if np.sum(obj_mask) < MIN_PIXEL_COUNT:
                        continue
                    valid_object_count += 1 
                    if gsd is not None:
                        object_area = np.sum(obj_mask) * (gsd ** 2)
                        rows, cols = np.where(obj_mask)
                        width_pixels = cols.max() - cols.min() + 1
                        height_pixels = rows.max() - rows.min() + 1
                        physical_width = width_pixels * gsd
                        physical_height = height_pixels * gsd

                        self.object_physical_areas[class_id].append(object_area)
                        self.object_dimensions[class_id].append((physical_width, physical_height))

                    y_center, x_center = map(int, ndimage.center_of_mass(obj_mask))
                    y_region = 'arriba' if y_center < h_third else 'medio' if y_center < 2 * h_third else 'abajo'
                    x_region = 'izquierda' if x_center < w_third else 'centro' if x_center < 2 * w_third else 'derecha'

                    region_key = f'{y_region}_{x_region}'
                    self.spatial_stats[region_key] += 1

                self.class_counts[class_id] += valid_object_count
                objects_this_image += valid_object_count


        if gsd is not None:
            density = objects_this_image / (img_area / 1e6)  
        else:
            density = objects_this_image/ (img_area / 1e6)
        if density > 10:
            self.area_classifications['Muy Densa'] += 1
        elif density > 5:
            self.area_classifications['Densa'] += 1
        else:
            self.area_classifications['Dispersa'] += 1
        
        self.total_objects += objects_this_image

        self.save_density_visualization(prediction, output_dir, img_name)

    def save_statistics(self, output_dir):
        stats_file = os.path.join(output_dir, 'class_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("Estadísticas de detección de objetos\n\n")
            f.write("-" * 40 + "\n")
            
            present_classes = [(i, count) for i, count in enumerate(self.class_counts[1:], start=1) if count > 0]
            sorted_classes = sorted([(i, count) for i, count in enumerate(self.class_counts[1:], start=1) 
                               if count > 0], key=lambda x: x[1], reverse=True)
            
            for class_id, count in sorted_classes:
                percentage = (count / self.total_objects * 100) if self.total_objects > 0 else 0
                f.write(f"{self.class_names[class_id]}:\n")
                f.write(f"Objetos detectados: {count:,}\n")
                f.write(f"Porcentaje: {percentage:.2f}%\n\n")
                
                areas = self.object_physical_areas.get(class_id, [])
                if areas:
                    avg_area = np.mean(areas)
                    min_area = np.min(areas)
                    max_area = np.max(areas)
                    std_area = np.std(areas)
                    f.write("Área física (m²):\n")
                    f.write(f"    Media: {avg_area:.2f}, Min: {min_area:.2f}, Max: {max_area:.2f}, Desviación estandar: {std_area:.2f}\n")

                dimensions = self.object_dimensions.get(class_id, [])
                if dimensions:
                    dimensions_arr = np.array(dimensions)
                    avg_width, avg_height = dimensions_arr.mean(axis=0)
                    min_width, min_height = dimensions_arr.min(axis=0)
                    max_width, max_height = dimensions_arr.max(axis=0)
                    f.write("  Dimensiones (m):\n")
                    f.write(f"    Media: anchura: {avg_width:.2f}, altura: {avg_height:.2f}\n")
                    f.write(f"    Min: anchura: {min_width:.2f}, altura: {min_height:.2f}\n")
                    f.write(f"    Max: anchura: {max_width:.2f}, altura: {max_height:.2f}\n")

                f.write("\n")
            
            f.write("Objetos mas comunes:\n")
            for class_id, count in sorted_classes[:3]:
                percentage = (count / self.total_objects * 100)
                f.write(f"- {self.class_names[class_id]}: {count} ({percentage:.1f}%)\n")
            f.write("\n")

            f.write("\nDensidad de área\n")
            f.write("-" * 40 + "\n\n")
            total_areas = sum(self.area_classifications.values())
            for density, count in self.area_classifications.items():
                percentage = (count / total_areas * 100)
                f.write(f"{density}: {percentage:.1f}%\n")
            
            f.write("\nDistribución espacial\n")
            f.write("-" * 40 + "\n\n")
            total_objects = sum(self.spatial_stats.values())
            for region, count in self.spatial_stats.items():
                percentage = (count / total_objects * 100) if total_objects > 0 else 0
                f.write(f"{region.replace('_', ' ').title()}: {count} objetos ({percentage:.1f}%)\n")

class RunEval():
    def __init__(self, output_dir, metrics, with_f1, write_image, dataset_cls, inference_mode, beta=1):
        self.output_dir = output_dir
        self.rgb_path = os.path.join(output_dir, 'rgb')
        self.pred_path = os.path.join(output_dir, 'pred')
        self.diff_path = os.path.join(output_dir, 'diff')
        self.compose_path = os.path.join(output_dir, 'compose')
        self.metrics = metrics
        self.with_f1 = with_f1
        self.beta = beta

        self.write_image = write_image
        self.dataset_cls = dataset_cls
        self.inference_mode = inference_mode
        self.time_list = []
        self.mapping = {}
        self.stats_tracker = StatisticsTracker(self.dataset_cls.num_classes, class_names=self.dataset_cls.class_names)
        os.makedirs(self.rgb_path, exist_ok=True)
        os.makedirs(self.pred_path, exist_ok=True)
        os.makedirs(self.diff_path, exist_ok=True)
        os.makedirs(self.compose_path, exist_ok=True)

        if self.metrics:
            self.hist = np.zeros((self.dataset_cls.num_classes,
                                  self.dataset_cls.num_classes))
            if self.with_f1:
                self.total_f1 = np.zeros((self.dataset_cls.num_classes, ), dtype=np.float)
        else:
            self.hist = None

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def inf(self, imgs, img_names, inference, net, scales, pbar, base_img, gt=None, gsd=None):

        ######################################################################
        # Run inference
        ######################################################################

        self.img_name = img_names[0]
        col_img_name = '{}/{}_color.png'.format(self.rgb_path, self.img_name)
        pred_img_name = '{}/{}.png'.format(self.pred_path, self.img_name)
        diff_img_name = '{}/{}_diff.png'.format(self.diff_path, self.img_name)
        compose_img_name = '{}/{}_compose.png'.format(self.compose_path, self.img_name)
        to_pil = transforms.ToPILImage()
        if self.inference_mode == 'pooling':
            img = imgs
            pool_base_img = to_pil(base_img[0])
        else:
            img = to_pil(imgs[0])
        prediction_pre_argmax_collection = inference(net, img, scales)

        if self.inference_mode == 'pooling':
            prediction = prediction_pre_argmax_collection
            prediction = np.concatenate(prediction, axis=0)[0]
        else:
            prediction_pre_argmax = np.mean(prediction_pre_argmax_collection, axis=0)
            prediction = np.argmax(prediction_pre_argmax, axis=0)

        if self.metrics and gt is not None:
            self.hist += fast_hist(prediction.flatten(), gt.cpu().numpy().flatten(),
                                   self.dataset_cls.num_classes)
            iou = round(np.nanmean(per_class_iu(self.hist)) * 100, 2)
            pbar.set_description("Mean IOU: %s" % (str(iou)))
        
        self.stats_tracker.update(prediction, self.img_name, self.output_dir, gsd=gsd[0])

        ######################################################################
        # Dump Images
        ######################################################################
        if self.write_image:

            if self.inference_mode == 'pooling':
                img = pool_base_img

            colorized = self.dataset_cls.colorize_mask(prediction)
            colorized.save(col_img_name)
            blend = Image.blend(img.convert("RGBA"), colorized.convert("RGBA"), 0.5)
            blend.save(compose_img_name)

            if gt is not None and args.split != 'test':
                gt = gt[0].cpu().numpy()
                # only write diff image if gt is valid
                diff = (prediction != gt)
                diff[gt == 255] = 0
                diffimg = Image.fromarray(diff.astype('uint8') * 255)
                PIL.ImageChops.lighter(
                    blend,
                    PIL.ImageOps.invert(diffimg).convert("RGBA")
                ).save(diff_img_name)

            label_out = np.zeros_like(prediction)
            for label_id, train_id in self.dataset_cls.label2trainid.items():
                label_out[np.where(prediction == train_id)] = label_id
            cv2.imwrite(pred_img_name, label_out)

    def final_dump(self):
        """
        Dump Final metrics on completion of evaluation
        """
        if self.metrics:
            evaluate_eval_for_inference(self.hist, args.dataset_cls, self.with_f1, beta=self.beta)
        self.stats_tracker.save_statistics(self.output_dir)



def infer_args():
    """
    To make life easier, we infer some args from the snapshot meta information.
    """
    if 'dist_bn' in args.snapshot and not args.dist_bn:
        args.dist_bn = True

    cv_re = re.search(r'-cv_(\d)-', args.snapshot)
    if cv_re and args.cv_split is None:
        args.cv_split = int(cv_re.group(1))

    snap_dir, _snap_file = os.path.split(args.snapshot)
    exp_dir, snap_dir = os.path.split(snap_dir)
    ckpt_path, exp_dir = os.path.split(exp_dir)
    ckpt_path = os.path.basename(ckpt_path)

    if args.exp_name is None:
        args.exp_name = exp_dir

    if args.ckpt_path is None:
        args.ckpt_path = ckpt_path

    if args.dataset == 'video_folder':
        args.split = 'video_folder'


def main():
    """
    Main Function
    """
    # Parse args and set up logging
    infer_args()

    if args.single_scale:
        scales = [1.0]
    else:
        scales = [float(x) for x in args.scales.split(',')]

    output_dir = os.path.join(args.ckpt_path, args.exp_name, args.split)
    os.makedirs(output_dir, exist_ok=True)
    save_log('eval', output_dir, date_str)
    logging.info("Network Arch: %s", args.arch)
    logging.info("CV split: %d", args.cv_split)
    logging.info("Exp_name: %s", args.exp_name)
    logging.info("Ckpt path: %s", args.ckpt_path)
    logging.info("Scales : %s", ' '.join(str(e) for e in scales))
    logging.info("Inference mode: %s", args.inference_mode)

    # Set up network, loader, inference mode
    metrics = args.dataset != 'video_folder'
    test_loader = setup_loader()

    runner = RunEval(output_dir, metrics,
                     write_image=args.dump_images,
                     dataset_cls=args.dataset_cls,
                     inference_mode=args.inference_mode,
                     with_f1=args.with_f1,
                     beta=args.beta)
    net = get_net()

    # Fix the ASPP pool size to 105, which is the tensor size if you train with crop
    # size of 840x840
    if args.fixed_aspp_pool:
        net.module.aspp.img_pooling = torch.nn.AvgPool2d(105)

    if args.inference_mode == 'sliding':
        inference = inference_sliding
    elif args.inference_mode == 'pooling':
        inference = inference_pool
    elif args.inference_mode == 'whole':
        inference = inference_whole
    else:
        raise 'Not a valid inference mode: {}'.format(args.inference_mode)

    # Run Inference!
    pbar = tqdm(test_loader, desc='eval {}'.format(args.split), smoothing=1.0)
    for iteration, data in enumerate(pbar):
        if args.dataset == 'video_folder':
            imgs, img_names = data
            gt = None
        else:
            if args.inference_mode == 'pooling':
                base_img, gt_with_imgs, img_names = data
                base_img = base_img[0]
                imgs = gt_with_imgs[0]
                gt = gt_with_imgs[1] if args.split != 'test' else None
            else:
                base_img = None
                imgs, gt, gsds, img_names = data
                print(f'GSDs: {gsds}')
        runner.inf(imgs, img_names, inference, net, scales, pbar, base_img, gt, gsds)
        #if iteration > 0 and args.test_mode:
         #   break

    # Calculate final overall statistics
    runner.final_dump()


if __name__ == '__main__':
    main()