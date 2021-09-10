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

from util.unsorted_utils import get_matches_from_int_masks



def collate_fn(batch):
    images = tuple([data['image'] for data in batch])
    return torch.stack(images), batch


def augment_unreal_data(given_data):
    data = copy(given_data)  #<! TODO: is this necessary 
    aug = Augmentor(data['image'].shape[1:] , s=0.5)
    data['augmentor'] = aug
    keys_to_augment = ['image']
    keys_to_geometrically_augment = ['depth', 'mask', 'classmask', 'unique_mask']

    for key in keys_to_augment:
        data[key] = aug(data[key])
    for key in keys_to_geometrically_augment:
        if len(data[key].shape) == 2: data[key] = data[key].unsqueeze(0)
        data[key] = aug.geometric_only(data[key])
        ########
        more_data = calculate_more_masks_for_whatever_reason(data)  # TODO: remove this or update it.
        data.update(more_data)  # TODO: remove like above
        ########
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
        dataset = Unreal_parts(args.data_dir, split, args.image_type, args.obj_class, args.n_pair, args.n_nonpair_singleobj, args.n_nonpair_bg)
    else:
        raise Exception("Unrecognized dataset {}".format(args.dataset))
    
    collect_func = collate_pair_of_augments
    
    if return_dataset: return dataset
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                         pin_memory=True, drop_last=True, collate_fn=collect_func)

def sample_from_mask(mask, k, blur=None):
    if blur is not None:
        assert(len(blur) == 2)
        # TODO: blur mask using a gaussian kernel
    mask_indices = (mask.squeeze() == 1).nonzero()
    k = min(k, mask_indices.shape[0])
    samples = random.sample(mask_indices.numpy().tolist(), k)
    return torch.tensor(samples).T

def sample_non_matches(sampled_indices, sigma, limits=None):
    zeros = torch.zeros_like(sampled_indices.float())
    offsets = torch.normal(zeros, sigma).long()
    N = sampled_indices + offsets
    print(N.shape)
    if limits is not None:
        limits = torch.tensor(limits).long()
        N = N.where(N[:] > 0, torch.tensor(0).long())
        N[0] = N[0].where(N[0] < limits[0], limits[0]-1)
        N[1] = N[1].where(N[1] < limits[1], limits[1]-1)
    return torch.reshape(N, (2, -1))
    

class Unreal_parts(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train', input_mode='RGBD', obj_class='mug', n_pair=1000, n_nonpair_singleobj=1000, n_nonpair_bg=1000):
        # read all rgb file list
        self.root = data_dir
        self.split = split        

        self.__compile_list_of_scenes()
        self.__load_camera_settings()

        self.input_mode = input_mode
        self.n_pair = n_pair
        self.n_nonpair_singleobj = n_nonpair_singleobj
        self.n_nonpair_bg = n_nonpair_bg


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
        sampled_scene = random.sample(self.scene_list, 1)[0]
        sampled_file = random.sample(self.rgb_filelist[sampled_scene], 1)[0]
        data  = self.get_data_as_dictionary(sampled_file)
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






# Has not been inspected
def get_mask_of_all_objects(data):
    point_masks = {}
    for obj in data['objmaskid']:
        uv0 = [torch.where(data['mask'] == id) for id in data['objmaskid'][obj]]
        if min([x[0].size(0) for x in uv0]) > 0:
            point_masks[obj] = torch.stack([torch.cat([x[1] for x in uv0]), torch.cat([x[0] for x in uv0])])
    return point_masks


def calculate_more_masks_for_whatever_reason(data, n_nonpair_singleobj=1000, n_nonpair_bg=1000):
    point_pairs = {}
    point_masks = get_mask_of_all_objects(data)
    point_pairs['fg_mask'] = point_masks
    
    if point_masks != {}:
        n_fg_pixels = sum([index.size(1) for _, index in point_masks.items()])
        # random foreground <-> background pixels
        all_bg_index = torch.flip(torch.stack(torch.where(data['classmask'] == 0)), [0])
        fg_index = []
        n_fg_pixels = sum([index.size(1) for _, index in point_masks.items()])
        for _, index in point_masks.items():
            rand_index = torch.randperm(index.size(1))[:index.size(1) * n_nonpair_bg // n_fg_pixels]
            fg_index.append(index[:, rand_index])
        fg_index = torch.cat(fg_index, dim=1)
        if fg_index.size(1) < n_nonpair_bg:
            fg_index = torch.cat([fg_index, fg_index[:, fg_index.size(1) - n_nonpair_bg:]], dim=1)
        rand_index = torch.randperm(all_bg_index.size(1))[:fg_index.size(1)]
        bg_index = all_bg_index[:, rand_index]
        point_pairs['nonpair_bg'] = torch.cat((fg_index, bg_index), dim=0)
        point_pairs['bg_mask'] = data['classmask'] == 0
        # random two foreground pixels on every single object
        single_index = []
        for _, index in point_masks.items():
            n_index = index.size(1) * n_nonpair_singleobj // n_fg_pixels
            rand_index1 = torch.randperm(index.size(1))[:n_index + 10] # leave some extra to remove repeating pairs
            rand_index2 = torch.randperm(index.size(1))[:n_index + 10]
            non_repeat_index = rand_index1 != rand_index2
            single_index.append(torch.cat([index[:, rand_index1[non_repeat_index][:n_index]], index[:, rand_index2[non_repeat_index][:n_index]]], dim=0))
        single_index = torch.cat(single_index, dim=1)
        if single_index.size(1) < n_nonpair_singleobj:
            single_index = torch.cat([single_index, single_index[:, single_index.size(1) - n_nonpair_singleobj:]], dim=1)
        point_pairs['nonpair_singleobj'] = single_index
    else:
        for name in ['nonpair_bg', 'nonpair_singleobj']:
            point_pairs[name] = torch.tensor([])
        point_pairs['bg_mask'] = torch.ones(data['classmask'].size(), dtype=torch.bool)
    return point_pairs