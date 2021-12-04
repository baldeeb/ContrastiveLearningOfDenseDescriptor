import os
import re
import cv2
import json
import glob
import torch
import random
import numpy as np
from copy import copy
from scipy.spatial.transform import Rotation as R
from augmentations.geometrically_invertible_aug import GeometricallyInvertibleAugmentation as Augmentor
from augmentations.util import union_of_augmented_images_in_original
from util.unsorted_utils import get_matches_from_int_masks
from util.sampling_utils import sample_from_mask, sample_non_matches

def collate_fn(batch):
    images = tuple([data['image'] for data in batch])
    return torch.stack(images), batch


def augment_unreal_data(given_data):
    data = copy(given_data)  #<! TODO: is this necessary 
    aug = Augmentor(data['image'].shape[1:])
    data['augmentor'] = aug
    keys_to_augment = ['image']
    keys_to_geometrically_augment = ['depth', 'mask', 'classmask', 'unique_mask']
    for key in keys_to_augment:
        data[key] = aug(data[key])
    for key in keys_to_geometrically_augment:
        if len(data[key].shape) == 2: data[key] = data[key].unsqueeze(0)
        data[key] = aug.geometric_only(data[key])
    return data


def collate_pair_of_augments(batch):
    num_augs = 2 # TODO: make this a functor init param
    new_batch = []
    for original_data in batch:
        shape = original_data['image'].shape[1:] 
        original_data['unique_mask'] = torch.arange(1, shape[0]*shape[1]+1).reshape((shape))
        pair = [augment_unreal_data(original_data) for i in range(num_augs)] #TODO: return pairs
        
        new_batch.extend(pair)
    return collate_fn(new_batch)


def make_data_loader(split, args, return_dataset=False):
    if args.dataset == "unreal_parts":
        dataset = Unreal_parts(args.data_dir, args.image_type)
    else:
        raise Exception("Unrecognized dataset {}".format(args.dataset))
    
    collect_func = collate_pair_of_augments
    
    if return_dataset: return dataset
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                         pin_memory=True, drop_last=True, collate_fn=collect_func)


class Unreal_parts(torch.utils.data.Dataset):
    def __init__(self, data_dir, input_mode='RGBD', randomly_index=True):
        # read all rgb file list
        self.root = data_dir 
        self.__compile_list_of_scenes()
        self.__load_camera_settings()
        self.input_mode = input_mode
        self.randomly_index = randomly_index

    def __load_camera_settings(self):
        data_path = os.path.abspath(os.path.join(self.root, 'Room_Capturer/_camera_settings.json'))
        cam_settings = json.load(open(data_path))
        i = cam_settings['camera_settings'][0]['intrinsic_settings']
        self.K = torch.tensor([[i['fx'], 0.0, i['cx']],
                                [0.0, i['fy'], i['cy']],
                                [0.0,   0.0,   1.0]])
        self.K_inv = torch.tensor([ [1/i['fx'], 0.0,        -i['cx']/i['fx']],
                                    [0.0,       1/i['fy'],  -i['cy']/i['fy']],
                                    [0.0,       0.0,        1.0]])
        im_size = cam_settings['camera_settings'][0]['captured_image_size']
        self.width, self.height = im_size['width'], im_size['height']
    
    def __compile_list_of_scenes(self):
        self.rgb_filelist = {'all': []}
        for scene_folder in os.listdir(self.root):
            self.rgb_filelist[scene_folder] = []
            for json_file in glob.glob(self.root + '/' + scene_folder + '/0*.json'):
                self.rgb_filelist[scene_folder].append(json_file.replace('.json', '.png'))
            self.rgb_filelist['all'] += self.rgb_filelist[scene_folder]
        self.rgb_filelist = {k: v for k, v in self.rgb_filelist.items() if v}
        self.scene_list = list(self.rgb_filelist)
        self.scene_list.remove('all')

    def __len__(self):
        return len(self.rgb_filelist['all'])
    
    def __getitem__(self, index):
        if self.randomly_index :
            sampled_scene = random.sample(self.scene_list, 1)[0]
            sampled_file = random.sample(self.rgb_filelist[sampled_scene], 1)[0]
        else :
            sampled_file = self.rgb_filelist['all'][index]
        
        data  = self.get_data_as_dictionary(sampled_file)
        data['index'] = index
        if self.input_mode == 'RGB':
            image = data['rgb'].permute(2, 0, 1)
        elif self.input_mode == 'RGBD':
            image = torch.cat((data['rgb'], data['depth']), dim=2).permute(2, 0, 1)
        elif self.input_mode == 'RGB-SurfaceNormal':
            surface_normal = self.depth2normal(data['depth'])
            image = torch.cat((data['rgb'], surface_normal), dim=2).permute(2, 0, 1)
        data['image'] = image
        return data

    def get_data_as_dictionary(self, rgbfile):
        depthfile = rgbfile.replace('.png', '.depth.mm.16.png')
        jsonfile = rgbfile.replace('.png', '.json')
        maskfile = rgbfile.replace('.png', '.is.png')
        classmaskfile = rgbfile.replace('.png', '.cs.png')

        data = {}
        data['src'] = rgbfile
        data['rgb'] = torch.tensor(np.array(cv2.imread(rgbfile)).astype(np.float32) / 255, dtype=torch.float)
        data['depth'] = torch.tensor(np.array(cv2.imread(depthfile, -1)).astype(np.float32) / 10, dtype=torch.float) # in centimeter
        mask = np.array(cv2.imread(maskfile)).astype(np.longlong)
        mask = mask[:, :, 2] * 256 * 256 + mask[:, :, 1] * 256 + mask[:, :, 0]
        data['mask'] = torch.tensor(mask, dtype=torch.long)
        data['classmask'] = torch.tensor(cv2.imread(classmaskfile)[:, :, 0], dtype=torch.long)
        with open(jsonfile, 'r') as f:
            meta = json.load(f)
        # group parts into objects by their class name in metadata
        obj2maskid, obj2pose, obj2cuboid = {}, {}, {}
        for obj in meta['objects']:
            part_id = re.findall(r'\d+', obj['class'].split('_')[1])
            obj_id = '1' if part_id == [] else part_id[0]
            obj_name = obj['class'].split('_')[0] + '_' + obj_id
            if obj_name not in obj2maskid.keys():
                obj2maskid[obj_name] = [obj['instance_id']]
                pose = torch.tensor(obj['pose_transform'])
                pose[:3, :3] = pose[:3, :3] / torch.norm(pose[:3, :3], dim=0) # rotation part needs normalization
                obj2pose[obj_name] = pose
                obj2cuboid[obj_name] = torch.tensor(obj['cuboid'])
            else:
                obj2maskid[obj_name].append(obj['instance_id'])
        data['objmaskid'] = obj2maskid
        data['objpose'] = obj2pose
        data['objcuboid'] = obj2cuboid
        return data


    def get_projected_indices(self, data):
        w, h = self.width, self.height
        
        # Create matrix of homogeneous pixel coordinates
        ui = torch.arange(0, w).unsqueeze(0).repeat(h, 1)
        vi = torch.arange(0, h).unsqueeze(1).repeat(1, w)
        ones = torch.ones((h, w))
        uv = torch.stack([ui, vi, ones], dim=0).type(torch.float)
        
        # Project to camera 3D frame
        d = data['depth'].unsqueeze(0)
        scaled_coords = uv * d.repeat(3, 1, 1)
        xyz = self.K_inv @ scaled_coords.view(3, -1)
        
        return xyz