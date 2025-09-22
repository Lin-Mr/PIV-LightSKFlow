import numpy as np
import torch
import torch.utils.data as data

import os
import random
from glob import glob
import os.path as osp

from utils import frame_utils
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.extra_info = []
        self.occ_list = None
        self.seg_list = None
        self.seg_inv_list = None

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        valid = None
        if self.sparse:
            flow, valid = frame_utils.readFlowKITTI(self.flow_list[index])
        else:
            flow = frame_utils.read_gen(self.flow_list[index])

        if self.occ_list is not None:
            occ = frame_utils.read_gen(self.occ_list[index])
            occ = np.array(occ).astype(np.uint8)
            occ = torch.from_numpy(occ // 255).bool()

        if self.seg_list is not None:
            f_in = np.array(frame_utils.read_gen(self.seg_list[index]))
            seg_r = f_in[:, :, 0].astype('int32')
            seg_g = f_in[:, :, 1].astype('int32')
            seg_b = f_in[:, :, 2].astype('int32')
            seg_map = (seg_r * 256 + seg_g) * 256 + seg_b
            seg_map = torch.from_numpy(seg_map)

        if self.seg_inv_list is not None:
            seg_inv = frame_utils.read_gen(self.seg_inv_list[index])
            seg_inv = np.array(seg_inv).astype(np.uint8)
            seg_inv = torch.from_numpy(seg_inv // 255).bool()

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)

        if self.occ_list is not None:
            return img1, img2, flow, valid.float(), occ, self.occ_list[index]
        elif self.seg_list is not None and self.seg_inv_list is not None:
            return img1, img2, flow, valid.float(), seg_map, seg_inv
        else:
            return img1, img2, flow, valid.float()#, self.extra_info[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)

class UnsupervisedFlowDataset(data.Dataset):
    def __init__(self, aug_params=None):
        self.augmentor = None
        if aug_params is not None:
            self.augmentor = FlowAugmentor(**aug_params)
            
        self.image_list = []
        self.init_seed = False

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        # 只加载图像对
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # 转换为RGB格式
        if len(img1.shape) == 2:  # 灰度图
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        # 数据增强
        if self.augmentor is not None:
            # 创建虚拟光流(全零)用于增强
            h, w = img1.shape[:2]
            dummy_flow = np.zeros((h, w, 2), dtype=np.float32)
            img1, img2, _ = self.augmentor(img1, img2, dummy_flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()

        # 返回图像对和虚拟valid mask(全1)
        valid = torch.ones(img1.shape[1], img1.shape[2], dtype=torch.float32)
        return img1, img2, valid

    def __len__(self):
        return len(self.image_list)



# class MpiSintel(FlowDataset):
#     def __init__(self, aug_params=None, split='training', root='datasets/Sintel', dstype='clean',
#                 occlusion=False, segmentation=False):
#         super(MpiSintel, self).__init__(aug_params)
#         flow_root = osp.join(root, split, 'flow')
#         image_root = osp.join(root, split, dstype)
#         occ_root = osp.join(root, split, 'occlusions')
#         # occ_root = osp.join(root, split, 'occ_plus_out')
#         # occ_root = osp.join(root, split, 'in_frame_occ')
#         # occ_root = osp.join(root, split, 'out_of_frame')

#         seg_root = osp.join(root, split, 'segmentation')
#         seg_inv_root = osp.join(root, split, 'segmentation_invalid')
#         self.segmentation = segmentation
#         self.occlusion = occlusion
#         if self.occlusion:
#             self.occ_list = []
#         if self.segmentation:
#             self.seg_list = []
#             self.seg_inv_list = []

#         if split == 'test':
#             self.is_test = True

#         for scene in os.listdir(image_root):
#             image_list = sorted(glob(osp.join(image_root, scene, '*.png')))
#             for i in range(len(image_list)-1):
#                 self.image_list += [ [image_list[i], image_list[i+1]] ]
#                 self.extra_info += [ (scene, i) ] # scene and frame_id

#             if split != 'test':
#                 self.flow_list += sorted(glob(osp.join(flow_root, scene, '*.flo')))
#                 if self.occlusion:
#                     self.occ_list += sorted(glob(osp.join(occ_root, scene, '*.png')))
#                 if self.segmentation:
#                     self.seg_list += sorted(glob(osp.join(seg_root, scene, '*.png')))
#                     self.seg_inv_list += sorted(glob(osp.join(seg_inv_root, scene, '*.png')))


# class FlyingChairs(FlowDataset):
#     def __init__(self, aug_params=None, split='training', root='datasets/FlyingChairs_release/data'):
#         super(FlyingChairs, self).__init__(aug_params)

#         images = sorted(glob(osp.join(root, '*.ppm')))
#         flows = sorted(glob(osp.join(root, '*.flo')))
#         assert (len(images)//2 == len(flows))
        
#         split_list = np.loadtxt('chairs_split.txt', dtype=np.int32)
#         for i in range(len(flows)):
#             xid = split_list[i]
#             if (split=='training' and xid==1) or (split=='validation' and xid==2):
#                 self.flow_list += [ flows[i] ]
#                 self.image_list += [ [images[2*i], images[2*i+1]] ]


# class FlyingThings3D(FlowDataset):
#     def __init__(self, aug_params=None, root='datasets/FlyingThings3D', split='training', dstype='frames_cleanpass'):
#         super(FlyingThings3D, self).__init__(aug_params)

#         if split == 'training':
#             for cam in ['left']:
#                 for direction in ['into_future', 'into_past']:
#                     image_dirs = sorted(glob(osp.join(root, dstype, 'TRAIN/*/*')))
#                     image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

#                     flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TRAIN/*/*')))
#                     flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

#                     for idir, fdir in zip(image_dirs, flow_dirs):
#                         images = sorted(glob(osp.join(idir, '*.png')) )
#                         flows = sorted(glob(osp.join(fdir, '*.pfm')) )
#                         for i in range(len(flows)-1):
#                             if direction == 'into_future':
#                                 self.image_list += [ [images[i], images[i+1]] ]
#                                 self.flow_list += [ flows[i] ]
#                             elif direction == 'into_past':
#                                 self.image_list += [ [images[i+1], images[i]] ]
#                                 self.flow_list += [ flows[i+1] ]

#         elif split == 'validation':
#             for cam in ['left']:
#                 for direction in ['into_future', 'into_past']:
#                     image_dirs = sorted(glob(osp.join(root, dstype, 'TEST/*/*')))
#                     image_dirs = sorted([osp.join(f, cam) for f in image_dirs])

#                     flow_dirs = sorted(glob(osp.join(root, 'optical_flow/TEST/*/*')))
#                     flow_dirs = sorted([osp.join(f, direction, cam) for f in flow_dirs])

#                     for idir, fdir in zip(image_dirs, flow_dirs):
#                         images = sorted(glob(osp.join(idir, '*.png')))
#                         flows = sorted(glob(osp.join(fdir, '*.pfm')))
#                         for i in range(len(flows) - 1):
#                             if direction == 'into_future':
#                                 self.image_list += [[images[i], images[i + 1]]]
#                                 self.flow_list += [flows[i]]
#                             elif direction == 'into_past':
#                                 self.image_list += [[images[i + 1], images[i]]]
#                                 self.flow_list += [flows[i + 1]]

#                 valid_list = np.loadtxt('things_val_test_set.txt', dtype=np.int32)
#                 self.image_list = [self.image_list[ind] for ind, sel in enumerate(valid_list) if sel]
#                 self.flow_list = [self.flow_list[ind] for ind, sel in enumerate(valid_list) if sel]

    
# class KITTI(FlowDataset):
#     def __init__(self, aug_params=None, split='training', root='datasets/kitti15'):
#         super(KITTI, self).__init__(aug_params, sparse=True)
#         if split == 'testing':
#             self.is_test = True

#         root = osp.join(root, split)
#         images1 = sorted(glob(osp.join(root, 'image_2/*_10.png')))
#         images2 = sorted(glob(osp.join(root, 'image_2/*_11.png')))

#         for img1, img2 in zip(images1, images2):
#             frame_id = img1.split('/')[-1]
#             self.extra_info += [ [frame_id] ]
#             self.image_list += [ [img1, img2] ]

#         if split == 'training':
#             self.flow_list = sorted(glob(osp.join(root, 'flow_occ/*_10.png')))


# class HD1K(FlowDataset):
#     def __init__(self, aug_params=None, root='datasets/hd1k'):
#         super(HD1K, self).__init__(aug_params, sparse=True)

#         seq_ix = 0
#         while 1:
#             flows = sorted(glob(os.path.join(root, 'hd1k_flow_gt', 'flow_occ/%06d_*.png' % seq_ix)))
#             images = sorted(glob(os.path.join(root, 'hd1k_input', 'image_2/%06d_*.png' % seq_ix)))

#             if len(flows) == 0:
#                 break

#             for i in range(len(flows)-1):
#                 self.flow_list += [flows[i]]
#                 self.image_list += [ [images[i], images[i+1]] ]

#             seq_ix += 1

class PIVDataset(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='PIV_datasets'):
        super(PIVDataset, self).__init__(aug_params)
        
        # 根据split选择对应的列表文件
        list_file = 'FlowData_train.list' if split == 'training' else 'FlowData_test.list'
        list_path = os.path.join(root, 'lists', list_file)

        # 从列表文件加载样本路径
        self.flow_list = []
        self.image_list = []
        
        with open(list_path, 'r') as f:
            for line in f:
                # 分割三个文件路径（制表符分隔）
                img1_path, img2_path, flow_path = line.strip().split('\t')
                
                # 验证文件名一致性
                base_prefix = os.path.basename(img1_path).split('_img1')[0]
                assert base_prefix == os.path.basename(img2_path).split('_img2')[0], \
                    f"Image pair mismatch: {img1_path} vs {img2_path}"
                assert base_prefix == os.path.basename(flow_path).split('_flow')[0], \
                    f"Flow mismatch: {img1_path} vs {flow_path}"
                
                self.flow_list.append(flow_path)
                self.image_list.append([img1_path, img2_path])


class PIVDataset_type(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='PIV_datasets'):
        super(PIVDataset_type, self).__init__(aug_params)
        
        # 根据split选择对应的列表文件
        list_file = 'FlowData_train_cylinder.list' if split == 'training' else 'FlowData_test_DNS_turbulence.list'
        list_path = os.path.join(root, 'lists', list_file)

        # 从列表文件加载样本路径
        self.flow_list = []
        self.image_list = []
        
        with open(list_path, 'r') as f:
            for line in f:
                # 分割三个文件路径（制表符分隔）
                img1_path, img2_path, flow_path = line.strip().split('\t')
                
                # 验证文件名一致性
                base_prefix = os.path.basename(img1_path).split('_img1')[0]
                assert base_prefix == os.path.basename(img2_path).split('_img2')[0], \
                    f"Image pair mismatch: {img1_path} vs {img2_path}"
                assert base_prefix == os.path.basename(flow_path).split('_flow')[0], \
                    f"Flow mismatch: {img1_path} vs {flow_path}"
                
                self.flow_list.append(flow_path)
                self.image_list.append([img1_path, img2_path])


class PIVDataset_single(UnsupervisedFlowDataset):
    def __init__(self, aug_params=None, split='training', root='/home/lin/PIV-SKFlow-main/PIV_datasets'):
        super(PIVDataset_single, self).__init__(aug_params)

        if split == 'training':
            # 保持root不变
            pass
        else:
            root += '/test/uniform'

        # 只加载图像文件
        images = sorted(glob(osp.join(root, '*.tif')))
        
        # 确保图像数量是偶数（成对）
        assert len(images) % 2 == 0, "图像数量必须是偶数"
        
        # 初始化空列表
        self.image_list = []
        self.flow_list = []  # 保持空列表
        
        # 按图像对组织数据
        for i in range(len(images) // 2):
            self.image_list.append([images[2*i], images[2*i+1]])
            # 没有对应的光流文件，保持flow_list为空或添加None
            # self.flow_list.append(None)  # 如果需要占位符
            
        print(f"加载了 {len(self.image_list)} 个图像对")


            
# def fetch_dataloader(args, TRAIN_DS='C+T+K/S'):
#     """ Create the data loader for the corresponding training set """

#     if args.stage == 'chairs':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
#         train_dataset = FlyingChairs(aug_params, split='training')
    
#     elif args.stage == 'things':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.4, 'max_scale': 0.8, 'do_flip': True}
#         clean_dataset = FlyingThings3D(aug_params, dstype='frames_cleanpass', split='training')
#         final_dataset = FlyingThings3D(aug_params, dstype='frames_finalpass', split='training')
#         train_dataset = clean_dataset + final_dataset

#     elif args.stage == 'sintel':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
#         things = FlyingThings3D(aug_params, dstype='frames_cleanpass')
#         sintel_clean = MpiSintel(aug_params, split='training', dstype='clean')
#         sintel_final = MpiSintel(aug_params, split='training', dstype='final')        

#         if TRAIN_DS == 'C+T+K+S+H':
#             kitti = KITTI({'crop_size': args.image_size, 'min_scale': -0.3, 'max_scale': 0.5, 'do_flip': True})
#             hd1k = HD1K({'crop_size': args.image_size, 'min_scale': -0.5, 'max_scale': 0.2, 'do_flip': True})
#             train_dataset = 100*sintel_clean + 100*sintel_final + 200*kitti + 5*hd1k + things

#         elif TRAIN_DS == 'C+T+K/S':
#             train_dataset = 100*sintel_clean + 100*sintel_final + things

#     elif args.stage == 'kitti':
#         aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.4, 'do_flip': False}
#         train_dataset = KITTI(aug_params, split='training')

#     train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
#            pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

#     print('Training with %d image pairs' % len(train_dataset))
#     return train_loader


def fetch_dataloader(args, TRAIN_DS='PIV'):
    """ Create the data loader for PIV training set """
    
    # 针对PIV数据的增强参数设置
    if args.stage == 'piv':
        aug_params = {
            'crop_size': args.image_size,      # 从参数获取裁剪尺寸
            'min_scale': -0.2,                # 适用于PIV的缩放范围
            'max_scale': 0.4,                 
            'do_flip': True,                  # 启用水平/垂直翻转
        }

        # 创建PIV数据集实例
        train_dataset = PIVDataset(
            aug_params=aug_params,
            split='training',
            root=args.piv_root               # 从参数获取数据集路径
        )



    elif args.stage == 'single':
 
         train_dataset = PIVDataset_single(
            split='training',
            root=args.piv_root               # 从参数获取数据集路径
        )
 



    # 创建数据加载器（统一配置）
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
           pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    
    print(f'训练集包含 {len(train_dataset)} 对图像')
    return train_loader
