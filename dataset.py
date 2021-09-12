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
    if not isinstance(sigma, list): sigma = [sigma]
    negatives = []
    zeros = torch.zeros_like(sampled_indices.float())
    for s in sigma:
        offsets = torch.normal(zeros, s).long()
        N = sampled_indices + offsets
        if limits is not None:
            limits = torch.tensor(limits).long()
            N = N.where(N[:] > 0, torch.tensor(0).long())
            N[0] = N[0].where(N[0] < limits[0], limits[0]-1)
            N[1] = N[1].where(N[1] < limits[1], limits[1]-1)
        negatives.append(N.reshape((2, -1)))
    if len(negatives) == 1: return negatives[0]
    else: return negatives
    
def union_of_augmented_images_in_original(images, augmentors):
    '''
    returns a mask of a common region in the original image where those images overlap
    '''
    sh = images.shape
    masks = torch.zeros((sh[0], sh[2], sh[3]))
    for i in range(sh[0]):
        inv = augmentors[i].geometric_inverse(images[i])
        collapsed = inv.sum(dim=0).squeeze()
        masks[i, collapsed!=0] = 1
    return torch.prod(masks, 0)

def sample_from_augmented_pair(images, augmentors, num_samples=2000):
    '''
    images are of shape NxCxHxW
    augmentors is a list of N invertible functions used to augment.
    '''
    limits = tuple(images.shape[2:4])
    mask = union_of_augmented_images_in_original(images, augmentors)
    samples = sample_from_mask(mask, num_samples)
    
    if len(samples) == 0: return mask, samples, samples
    
    non_matches = sample_non_matches(samples, sigma=[10, 25, 50], limits=limits)
    return mask, samples, non_matches

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


