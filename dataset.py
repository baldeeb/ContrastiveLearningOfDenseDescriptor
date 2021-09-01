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


def calculate_more_masks_for_whatever_reason(data, n_nonpair_singleobj=1000, n_nonpair_bg=1000):
    point_masks, point_pairs = {}, {}
    for obj in data['objmaskid']:
        # if self.obj_class in obj:
        uv0 = [torch.where(data['mask'] == id) for id in data['objmaskid'][obj]]
        if min([x[0].size(0) for x in uv0]) > 0:
            point_masks[obj] = torch.stack([torch.cat([x[1] for x in uv0]), torch.cat([x[0] for x in uv0])])
    point_pairs['fg_mask'] = point_masks
    
    if point_masks != {}:
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


# def collate_fn(batch):
#     img_a, img_b, targets = tuple(zip(*batch))
#     min_npair = float('inf')
#     for i in range(len(targets)):
#         if len(targets[i]['pair']) > 0:
#             min_npair = min(min_npair, targets[i]['pair'].size(1))
#         else:
#             min_npair = 0
#             break
#     for i in range(len(targets)):
#         if min_npair > 0:
#             targets[i]['pair'] = targets[i]['pair'][:, :min_npair]
#         else:
#             del targets[i]['pair']
#     return torch.stack(img_a), torch.stack(img_b), targets


def collate_fn(batch):
    images = tuple([data['image'] for data in batch])
    return torch.stack(images), batch


def collate_augments(batch):
    num_augs = 2 # TODO: make this a functor init param
    new_batch = []
    for original_data in batch:
        keys_to_augment = ['image']
        keys_to_geometrically_augment = ['depth', 'mask', 'classmask']
        for count in range(num_augs):
            data = copy(original_data)
            aug = Augmentor(data['image'].shape[1:])
            for key in keys_to_augment:
                data[key] = aug(data[key])
            for key in keys_to_geometrically_augment:
                if len(data[key].shape) == 2: data[key] = data[key].unsqueeze(0)
                data[key] = aug.geometric_only(data[key])
                ########
                more_data = calculate_more_masks_for_whatever_reason(data)  # TODO: remove this or update it.
                data.update(more_data)  # TODO: remove like above
                ########
            new_batch.append(data)        
    return collate_fn(new_batch)


def make_data_loader(split, args, return_dataset=False):
    if args.dataset == "unreal_parts":
        dataset = Unreal_parts(args.data_dir, split, args.input_mode, args.obj_class, args.n_pair, args.n_nonpair_singleobj, args.n_nonpair_bg)
    else:
        raise Exception("Unrecognized dataset {}".format(args.dataset))
    
    collect_func = collate_augments

    # if args.batch_size > 1:
    #     collect_func = collate_fn 
    # elif augment:
    #     collect_func = collate_and_augment
    # else:
    #     collect_func = None
    
    
    if return_dataset: return dataset
    return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                         pin_memory=True, drop_last=True, collate_fn=collect_func)


def sample_from_mask(mask, k, blur=None):
    if blur is not None:
        assert(len(blur) == 2)
        # TODO: blur mask using a gaussian kernel
    mask_indices = (mask.squeeze() == 1).nonzero()
    samples = random.sample(mask_indices.numpy().tolist(), k)
    return torch.tensor(samples).T

def sample_non_matches(sampled_indices, sigma):
    flattend = sampled_indices.flatten().float()
    non_matches = torch.normal(flattend, sigma)
    return torch.reshape(non_matches, (2, -1))
    

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
        # self.obj_class = obj_class
        # self.occlusion_margin = 0.3 # cm
        # self.n_point_threshold = 20

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
        
        ############################ TODO: temporary, remove!########################################################
        for f in [depthfile, jsonfile, maskfile, classmaskfile]:
            if not os.path.isfile(f):
                print(f"############################\n\nfile {f} was not found!!!!\n\n############################")
        ##############################################################################################################

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
            obj_name, part_id = obj['class'].split('_')[0], re.findall(r'\d+', obj['class'].split('_')[1])
            obj_id = '1' if part_id == [] else part_id[0]
            obj_name = obj_name + '_' + obj_id
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
    









########################################################Currently not used ####################################################################################


    @staticmethod
    def depth2normal(d_im):
        d_im = d_im.numpy().astype("float32")
        zy, zx = np.gradient(d_im)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        # zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=5)
        # zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=5)
        normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        # offset and rescale values to be in 0-1
        normal += 1
        normal /= 2
        return torch.tensor(normal)

if __name__ == '__main__':
    args = parse_args('train')
    args.data_dir = '/home/cxt/Documents/research/affordance/self_supervised_learning/data/unreal_unsupervised'
    dataloader = make_data_loader('train', args)
    for i, (img_a, img_b, targets) in enumerate(dataloader):
        pass



    # def calculate_match_nonmatch(self, data):
    # #     point_pair_perobj, point_pairs = {}, {}
    # #     # get all corresponding point pairs between two images through known relative pose projection
    # #     for obj in data[0]['objmaskid']:
    # #         if self.obj_class in obj and obj in data[1]['objmaskid']:
    # #             uv0 = [torch.where(data[0]['mask'] == id) for id in data[0]['objmaskid'][obj]]
    # #             u0, v0 = torch.cat([x[1] for x in uv0]), torch.cat([x[0] for x in uv0])
    # #             z0 = data[0]['depth'][v0, u0]
    # #             xyz0_normalized = torch.matmul(self.K_inv, torch.stack([u0, v0, torch.ones_like(u0)], dim=1).T.clone().detach().float())
    # #             xyz0 = torch.mul(xyz0_normalized, z0.repeat(3, 1))
    # #             xyz0_objframe = torch.matmul(torch.inverse(data[0]['objpose'][obj].T), torch.cat([xyz0, torch.ones_like(z0).view(1, -1)], dim=0))
    # #             xyz1 = torch.matmul(data[1]['objpose'][obj].T, xyz0_objframe)[:3]
    # #             xyz1_normalized = torch.div(xyz1, xyz1[2].repeat(3, 1))
    # #             uv1 = torch.matmul(self.K, xyz1_normalized)
    # #             u1, v1 = uv1[0], uv1[1]
    # #             inbound_index = torch.mul(torch.mul(u1 > 0, u1 < self.width), torch.mul(v1 > 0, v1 < self.height))
    # #             u0, v0, u1, v1, z1 = u0[inbound_index], v0[inbound_index], u1[inbound_index], v1[inbound_index], xyz1[2, inbound_index]
    # #             u0 = u0.clamp(0, self.width-1)
    # #             v0 = v0.clamp(0, self.height-1)
    # #             u1 = u1.clamp(0, self.width-1)
    # #             v1 = v1.clamp(0, self.height-1)
    # #             z1_obs = data[1]['depth'][v1.long(), u1.long()]
    # #             occlusion_index = torch.mul(z1_obs > z1, z1_obs < z1 + self.occlusion_margin)
    # #             # print('#points within distance threshold:', torch.sum(occlusion_index))
    # #             if torch.sum(occlusion_index) >= self.n_point_threshold:
    # #                 point_pair_perobj[obj] = torch.stack([u0, v0, u1, v1])[:, occlusion_index]
    #     # get object foreground mask in two images
    #     point_masks = [{}, {}]
    #     for i in range(2):
    #         for obj in data[i]['objmaskid']:
    #             if self.obj_class in obj:
    #                 uv0 = [torch.where(data[i]['mask'] == id) for id in data[i]['objmaskid'][obj]]
    #                 if min([x[0].size(0) for x in uv0]) > 0:
    #                     point_masks[i][obj] = torch.stack([torch.cat([x[1] for x in uv0]), torch.cat([x[0] for x in uv0])])
    #     point_pairs['fg_mask0'] = point_masks[0]
    #     point_pairs['fg_mask1'] = point_masks[1]
    # #     if point_pair_perobj != {}:
    # #         all_pairs = torch.cat([x for _, x in point_pair_perobj.items()], dim=1)
    # #         rand_fg_index = torch.randperm(all_pairs.size(1))[:self.n_pair]
    # #         point_pairs['pair'] = all_pairs[:, rand_fg_index] # random corresponding point pairs in img1, img2
    # #     else:
    # #         point_pairs['pair'] = torch.tensor([])

    #     for i in range(2):
    #         if point_masks[i] != {}:
    #             # random foreground <-> background pixels
    #             all_bg_index = torch.flip(torch.stack(torch.where(data[i]['classmask'] == 0)), [0])
    #             fg_index = []
    #             n_fg_pixels = sum([index.size(1) for _, index in point_masks[i].items()])
    #             for _, index in point_masks[i].items():
    #                 rand_index = torch.randperm(index.size(1))[:index.size(1) * self.n_nonpair_bg // n_fg_pixels]
    #                 fg_index.append(index[:, rand_index])
    #             fg_index = torch.cat(fg_index, dim=1)
    #             if fg_index.size(1) < self.n_nonpair_bg:
    #                 fg_index = torch.cat([fg_index, fg_index[:, fg_index.size(1) - self.n_nonpair_bg:]], dim=1)
    #             rand_index = torch.randperm(all_bg_index.size(1))[:fg_index.size(1)]
    #             bg_index = all_bg_index[:, rand_index]
    #             point_pairs['nonpair_bg' + str(i)] = torch.cat((fg_index, bg_index), dim=0)
    #             point_pairs['bg_mask' + str(i)] = data[i]['classmask'] == 0
    #             # random two foreground pixels on every single object
    #             single_index = []
    #             for _, index in point_masks[i].items():
    #                 n_index = index.size(1) * self.n_nonpair_singleobj // n_fg_pixels
    #                 rand_index1 = torch.randperm(index.size(1))[:n_index + 10] # leave some extra to remove repeating pairs
    #                 rand_index2 = torch.randperm(index.size(1))[:n_index + 10]
    #                 non_repeat_index = rand_index1 != rand_index2
    #                 single_index.append(torch.cat([index[:, rand_index1[non_repeat_index][:n_index]], index[:, rand_index2[non_repeat_index][:n_index]]], dim=0))
    #             single_index = torch.cat(single_index, dim=1)
    #             if single_index.size(1) < self.n_nonpair_singleobj:
    #                 single_index = torch.cat([single_index, single_index[:, single_index.size(1) - self.n_nonpair_singleobj:]], dim=1)
    #             point_pairs['nonpair_singleobj' + str(i)] = single_index
    #         else:
    #             for name in ['nonpair_bg', 'nonpair_singleobj']:
    #                 point_pairs[name + str(i)] = torch.tensor([])
    #             point_pairs['bg_mask' + str(i)] = torch.ones(data[i]['classmask'].size(), dtype=torch.bool)
    #     return point_pairs










  