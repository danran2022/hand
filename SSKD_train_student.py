import os
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os.path as osp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse
import time
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import scipy.io as sio
import logging
import numpy.linalg as LA
import torch.backends.cudnn as cudnn
import json
import abc
from torch.utils.data import DataLoader
import torch.optim
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
from progress.bar import Bar
import losses as losses
import utils.misc as misc
from datasets.egodexter import EgoDexter
from datasets.handataset import HandDataset
from model.detnet import detnet
# from model.detnet import net_2d,net_3d
from utils import func, align
from utils.eval.evalutils import AverageMeter, accuracy_heatmap
from utils.eval.zimeval import EvalUtil
from model.helper import resnet50, conv3x3
import random
import torch.utils.data
from PIL import Image, ImageFilter
from termcolor import colored
from tqdm import tqdm
import config as cfg
import utils.func as func
import utils.handutils as handutils
import utils.heatmaputils as hmutils
import utils.imgutils as imutils
from datasets.dexter_object import DexterObjectDataset
from datasets.ganerated_hands import GANeratedDataset
from datasets.hand143_panopticdb import Hand143_panopticdb
from datasets.hand_labels import Hand_labels

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
DEBUG = 0

from datasets.handataset import HandDataset

import numpy as np
def global_align(gtj0, prj0):
    gtj = gtj0.copy()
    prj = prj0.copy()
    # gtj :B*21*3
    # prj :B*21*3
    root_idx = 9  # root
    ref_bone_link = [0, 9]  # mid mcp
    pred_align = prj.copy()
    for i in range(prj.shape[0]):

        pred_ref_bone_len = np.linalg.norm(prj[i][ref_bone_link[0]] - prj[i][ref_bone_link[1]])
        gt_ref_bone_len = np.linalg.norm(gtj[i][ref_bone_link[0]] - gtj[i][ref_bone_link[1]])
        scale = gt_ref_bone_len / pred_ref_bone_len

        for j in range(21):
            pred_align[i][j] = gtj[i][root_idx] + scale * (prj[i][j] - prj[i][root_idx])

    return gtj, pred_align


import torch
import torch.nn.functional as F


def kl_divergence(teacher_output, student_output):
    num_heatmaps = teacher_output.size(1)
    # 应用 softmax 函数
    teacher_probs = F.softmax(teacher_output, dim=1)
    student_probs = F.softmax(student_output, dim=1)

    # 计算 KL 散度
    kl_loss = torch.sum(student_probs * (torch.log(student_probs) - torch.log(teacher_probs)), dim=1)
    kl_loss = torch.mean(kl_loss) / num_heatmaps  # 对每个样本的 KL 散度取平均

    return kl_loss


def rotate_clr(clr_no_changes, pose_gts, device):
    blur_radius = 0.5
    brightness = 0.5
    saturation = 0.5
    hue = 0.15
    contrast = 0.5

    images_changed = torch.randn(clr_no_changes.size())
    images_changed_nor = torch.randn(clr_no_changes.size()).permute(0, 3, 1, 2)
    rot_tensor = torch.randn(clr_no_changes.size(0))
    rot_mat_all = torch.randn(clr_no_changes.size(0), 3, 3)
    joints_changed = torch.randn(clr_no_changes.size(0), 21, 3)
    for idx in range(clr_no_changes.size(0)):
        max_rot = np.pi
        rng = np.random.RandomState(seed=random.randint(0, 1024))
        rot = rng.uniform(low=-max_rot, high=max_rot)
        rot_angle = torch.tensor(rot, dtype=torch.float64)
        rot_tensor[idx] = rot_angle

        rot_mat = np.array([
            [np.cos(rot), -np.sin(rot), 0],
            [np.sin(rot), np.cos(rot), 0],
            [0, 0, 1],
        ]).astype(np.float32)
        rot_mat_all[idx] = torch.tensor(rot_mat)

        clr_no_change_pil = TF.to_pil_image(clr_no_changes[idx].permute(2, 0, 1))
        rotated_clr_no_change_pil = TF.rotate(clr_no_change_pil, angle=(-rot_angle).item() * (180 / 3.14159))
        clr_changed = rotated_clr_no_change_pil.filter(ImageFilter.GaussianBlur(blur_radius))
        clr_changed = imutils.color_jitter(
            clr_changed,
            brightness=brightness,
            saturation=saturation,
            hue=hue,
            contrast=contrast,
        )

        clr_changed_array = np.array(clr_changed)
        images_changed[idx] = torch.tensor(clr_changed_array)

        image_tensor = func.to_tensor(clr_changed_array).float()
        image_tensor = func.normalize(image_tensor, [0.5, 0.5, 0.5], [1, 1, 1])
        images_changed_nor[idx] = image_tensor

        joint = pose_gts[idx].cpu().numpy()
        joint_changed = rot_mat.dot(
            joint.transpose(1, 0)
        ).transpose()
        joints_changed[idx] = torch.tensor(joint_changed)
        # rotated_clrs[idx] = torch.tensor(np.array(rotated_clr_no_change_pil))

    return images_changed.to(device), images_changed_nor.to(device), rot_tensor.to(device), rot_mat_all.to(
        device), joints_changed.to(device)


def SS_inverse_change(joints_changed, rot_mat_all, device):
    # 根据旋转矩阵，将旋转后的3D手势关键点逆变换
    joint_before_rotation = torch.randn([joints_changed.size(0), 64, 2])
    for index_inverse in range(joints_changed.size(0)):
        joint_change = joints_changed[index_inverse].cpu()
        xy_rot_mat = rot_mat_all[index_inverse][:2, :2].cpu()
        rot_mat_inv = xy_rot_mat.T
        joint_before = torch.matmul(rot_mat_inv, joint_change.transpose(0, 1)).transpose(0, 1)
        joint_before_rotation[index_inverse] = joint_before

    return joint_before_rotation.to(device)

def vanila_contrastive_loss(z1, z2, temperature: float = 0.5) :
    """Calculates the contrastive loss as mentioned in SimCLR paper
        https://arxiv.org/pdf/2002.05709.pdf.
    Parts of the code adapted from pl_bolts nt_ext_loss.

    Args:
        z1 (Tensor): Tensor of normalized projections of the images.
            (#samples_in_batch x vector_dim).
        z2 (Tensor): Tensor of normalized projections of the same images but with
            different transformation.(#samples_in_batch x vector_dim)
        temperature (float, optional): Temperature term in the contrastive loss.
            Defaults to 0.5. In SimCLr paper it was shown t=0.5 is good for training
            with small batches.

    Returns:
        Tensor: Contrastive loss (1 x 1)
    """
    z = torch.cat([z1, z2], dim=0)
    n_samples = len(z)

    # Full similarity matrix
    cov = torch.mm(z, z.t().contiguous())
    sim = torch.exp(cov / temperature)

    # Negative similarity
    mask = ~torch.eye(n_samples, device=sim.device).bool()
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

    # Positive similarity :
    pos = torch.exp(torch.sum(z1 * z2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / neg).mean()
    return loss

batch_size=32

train_dataset = HandDataset(
    data_split='train',
    train=True,
    subset_name="stb",
    data_root="/home/media/ExtHDD1/zgw",
    scale_jittering=0.1,
    center_jettering=0.1,
    max_rot=0.5 * np.pi,
)

# /home/media/ExtHDD1/zgw
test_dataset = HandDataset(data_split='test',subset_name="stb",data_root="/home/media/ExtHDD1/zgw",train = False)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True, drop_last=False
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size,
    shuffle=False,
    num_workers=0
)

model = detnet()
model.to(device)

checkpoint_path_t = os.path.join("./checkpoints",'ckp_detnet_41.pth')
state_dict_t = torch.load(checkpoint_path_t)
# if args.clean:
state_dict = misc.clean_state_dict(state_dict_t)
model.load_state_dict(state_dict)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

# 加载 MobileNetV3 Small
mobilenet_v3_small_model = mobilenet_v3_small(pretrained=True)


# 修改 MobileNetV3 Small 作为学生网络
class MobileNetV3_small(nn.Module):
    def __init__(self):
        super(MobileNetV3_small, self).__init__()
        # 获取 MobileNetV3 Small 的 features 部分
        self.base_model = mobilenet_v3_small_model.features
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        # 添加一个卷积层来减少通道数
        self.reduce_channels = nn.Conv2d(576, 256, kernel_size=1)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.last_linear = nn.Linear(576, 21 * 3)

    def forward(self, x):
        x = self.base_model(x)
        feature = self.upsample(x)
        feature = self.reduce_channels(feature)

        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.last_linear(x)
        x = x.view(x.size(0), 21, 3)  # 调整输出形状为 20x3

        return feature, x

import torch
import torch.nn as nn
import torch.nn.functional as F

class Teacher(nn.Module):

    def __init__(self, module):
        super(Teacher, self).__init__()

        self.backbone = module
        # feat_dim = list(module.children())[-1].in_features
        #         self.projection_head = nn.Sequential(
        #                 nn.AdaptiveAvgPool2d(1),  # 添加池化层进行下采样
        #                 nn.Flatten(),  # 将池化后的特征图展平成一维张量
        #                 nn.Linear(256,128,bias=True,),
        #                 nn.BatchNorm1d(128),
        #                 nn.ReLU(),
        #                 nn.Linear(128,64,bias=False,),
        #             )
        #         self.projection_head = nn.Sequential(
        #                 nn.AdaptiveAvgPool2d(1),  # 添加池化层进行下采样
        #                 nn.Flatten(),  # 将池化后的特征图展平成一维张量
        #                 nn.Linear(256,128,bias=True,),
        #                 nn.BatchNorm1d(128),
        #                 nn.ReLU(),
        #             )
        self.projection_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4, stride=4, padding=1),
            nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=1),
            nn.Flatten(),  # 将池化后的特征图展平成一维张量
            nn.Linear(128 * 4, 128 * 2, bias=True, ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(128 * 2, 128, bias=False, ),
        )

    def forward(self, x, bb_grad=True):
        out = self.backbone(x)
        feats = out["features_mid"]
        if not bb_grad:
            feats = feats.detach()

        z = self.projection_head(feats)

        return out, z, feats


import torch
import torch.nn as nn
import torch.nn.functional as F


class Student(nn.Module):

    def __init__(self, module):
        super(Student, self).__init__()

        self.backbone = module
        # feat_dim = list(module.children())[-1].in_features
        #         self.projection_head = nn.Sequential(
        #                 nn.AdaptiveAvgPool2d(1),  # 添加池化层进行下采样
        #                 nn.Flatten(),  # 将池化后的特征图展平成一维张量
        #                 nn.Linear(256,128,bias=True,),
        #                 nn.BatchNorm1d(128),
        #                 nn.ReLU(),
        #                 nn.Linear(128,64,bias=False,),
        #             )
        #         self.projection_head = nn.Sequential(
        #                 nn.AdaptiveAvgPool2d(1),  # 添加池化层进行下采样
        #                 nn.Flatten(),  # 将池化后的特征图展平成一维张量
        #                 nn.Linear(256,128,bias=True,),
        #                 nn.BatchNorm1d(128),
        #                 nn.ReLU(),
        #             )
        self.projection_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4, stride=4, padding=1),
            nn.Conv2d(128, 128, kernel_size=4, stride=4, padding=1),
            nn.Flatten(),  # 将池化后的特征图展平成一维张量
            nn.Linear(128 * 4, 128 * 2, bias=True, ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(128 * 2, 128, bias=False, ),
        )

    def forward(self, x, bb_grad=True):
        feats, out = self.backbone(x)
        if not bb_grad:
            feats = feats.detach()

        z = self.projection_head(feats)

        return out, z, feats

teacher = Teacher(model).to(device)

student_net = MobileNetV3_small()
student_net_step1 = MobileNetV3_small()
student_net_eval = MobileNetV3_small()

# checkpoint = torch.load(f'./outputs/student_net_epoch_123.pth')
# student_net_step1.load_state_dict(checkpoint['model_state_dict'])

student_train_step1 = Student(student_net_step1).to(device)
student_train = Student(student_net).to(device)
student_eval = Student(student_net_eval).to(device)

student_train_step1.to(device)
student_train.to(device)
student_eval.to(device)
teacher.to(device)

# 计算模型大小并转换为MB
total_params = sum(p.numel() for p in teacher.parameters())
total_size = total_params / (1024 ** 2)
print(f"模型的总参数量为：{total_params},为{total_size:.2f} MB")

# 计算模型大小并转换为MB
total_params = sum(p.numel() for p in student_train.parameters())
total_size = total_params / (1024 ** 2)
print(f"模型的总参数量为：{total_params},为{total_size:.2f} MB")

# 在每个epoch循环外部初始化一个空列表来保存每个epoch的损失
epoch_loss_list = []
avg_est_error_list = []

loaded_data = np.load('./data/STB/data/min_max.npz')
min_values = loaded_data['array1']
max_values = loaded_data['array2']

# optimizer_step1 = torch.optim.Adam(filter(lambda p: p.requires_grad, student_net_step1.parameters()), lr=0.001)
optimizer = torch.optim.Adam(student_train.parameters(), lr=0.0001)
optimizer_step1 = torch.optim.Adam(student_train_step1.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
scheduler_step1 = StepLR(optimizer_step1, step_size=20, gamma=0.5)

mse_loss = torch.nn.MSELoss()
# alpha = 0.25
epochs = 200

# 将数据增强前图片作为原图片，将数据增强后图片作为对比学习对照图片
# 直接定义在out层上训练
for epoch in range(epochs):
    start = time.perf_counter()
    student_train.train()
    loss_list = []
    # 保存模型
    min_loss_index = -1
    min_loss = float('inf')
    r_t = 1 - (epoch / epochs)
    scheduler.step()

    for i, metas in enumerate(train_loader):
        images_nor = metas["clr"].to(device)
        images_pro = metas["clr_pro"].to(device)
        pose_gts_nor = metas["joint_nor"].to(device)
        pose_gts_pro = metas["joint_pro"].to(device)
        pose_gts = metas["joint"].to(device)
        clr_no_change = metas["clr_no_change"].to(device)
        clr_no_change_nor = metas["clr_no_change_nor"].to(device)
        joint_no_change_nor = metas["joint_no_change_nor"].to(device)
        # joint_no_change_nor = metas["joint_pro"].to(device)
        # clr_no_change，pose_gts_pro为数据增强前的图片和标签
        # clr_no_change_nor，joint_no_change_nor为数据增强前且归一化的图片和标签
        # images_nor，pose_gts_nor为数据增强后且归一化的图片和标签
        rot_mat_all = metas["rot_mat_all"].to(device)

        batch = images_nor.size(0)
        images_nor_combined = torch.cat((clr_no_change_nor, images_nor), dim=0)

        optimizer.zero_grad()
        out_S, z_S, feats_S = student_train(images_nor_combined, bb_grad=True)

        with torch.no_grad():
            out_T, z_T, feats_T = teacher(images_nor_combined, bb_grad=False)

        pose_gts_combined = torch.cat((pose_gts_pro, pose_gts), dim=0)

        loss_stu = mse_loss(out_S[:batch, :, :].to(torch.float32), joint_no_change_nor.to(torch.float32))

        feats_S_nor = feats_S[:batch, :, :, :]
        feats_S_aug = feats_S[batch:, :, :, :]
        feats_T_nor = feats_T[:batch, :, :, :]
        feats_T_aug = feats_T[batch:, :, :, :]

        # loss2
        loss_kd_fea_nor = mse_loss(feats_S_nor.to(torch.float32), feats_T_nor.to(torch.float32))

        # loss3
        loss_kd_fea_aug = mse_loss(feats_S_aug.to(torch.float32), feats_T_aug.to(torch.float32))

        # out_S_aug_inverse = SS_inverse_change(out_S_aug.view(out_S_aug.size(0), 21, 3), rot_mat_all, device)
        # out_S_aug_inverse = out_S_aug_inverse.view(out_S_aug_inverse.size(0), 21*3)
        # nor_rep_S = out_S_nor.view(out_S_nor.size(0), 21*3).unsqueeze(2).expand(-1,-1,batch).transpose(0,2)
        # aug_rep_S = out_S_aug_inverse.unsqueeze(2).expand(-1,-1,batch)
        # s_simi = F.cosine_similarity(aug_rep_S, nor_rep_S, dim=1)

        # out_T_aug_inverse = SS_inverse_change(out_T_aug.view(out_T_aug.size(0), 21, 3), rot_mat_all, device)
        # out_T_aug_inverse = out_T_aug_inverse.view(out_T_aug_inverse.size(0), 21*3)
        # nor_rep_T = out_T_nor.view(out_T_nor.size(0), 21*3).unsqueeze(2).expand(-1,-1,batch).transpose(0,2)
        # aug_rep_T = out_T_aug_inverse.unsqueeze(2).expand(-1,-1,batch)
        # t_simi = F.cosine_similarity(aug_rep_T, nor_rep_T, dim=1)

        # loss_SS_kd = F.mse_loss(s_simi, t_simi)
        z_S_nor = z_S[:batch, :]
        z_S_aug = z_S[batch:, :]
        z_S_aug_inverse = SS_inverse_change(z_S_aug.view(z_S_aug.size(0), 64, 2), rot_mat_all, device)
        z_S_aug_inverse = z_S_aug_inverse.view(z_S_aug_inverse.size(0), 64 * 2)
        nor_rep_S = z_S_nor.unsqueeze(2).expand(-1, -1, batch).transpose(0, 2)
        aug_rep_S = z_S_aug_inverse.unsqueeze(2).expand(-1, -1, batch)
        s_simi = F.cosine_similarity(aug_rep_S, nor_rep_S, dim=1)

        z_T_nor = z_T[:batch, :]
        z_T_aug = z_T[batch:, :]
        z_T_aug_inverse = SS_inverse_change(z_T_aug.view(z_T_aug.size(0), 64, 2), rot_mat_all, device)
        z_T_aug_inverse = z_T_aug_inverse.view(z_T_aug_inverse.size(0), 64 * 2)
        nor_rep_T = z_T_nor.unsqueeze(2).expand(-1, -1, batch).transpose(0, 2)
        aug_rep_T = z_T_aug_inverse.unsqueeze(2).expand(-1, -1, batch)
        t_simi = F.cosine_similarity(aug_rep_T, nor_rep_T, dim=1)

        loss_SS_kd = F.mse_loss(s_simi, t_simi)

        #         loss = 0.25 * loss_stu +0.75*(loss_kd_fea_nor + loss_kd_fea_nor + 10.0 * loss_SS_kd)
        loss = 10.0 * loss_stu + loss_kd_fea_nor + loss_kd_fea_nor + 10.0 * loss_SS_kd

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())  # 将每个batch的损失值添加到列表中

        print(
            f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()} , loss_stu: {loss_stu.item()}, loss_kd_fea_nor: {loss_kd_fea_nor.item()}, loss_kd_fea_aug: {loss_kd_fea_aug.item()},loss_SS_kd: {loss_SS_kd.item()}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': student_train.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 可选: 其他你想保存的内容
    }

    torch.save(checkpoint, f'./outputs/SSKD/3/S_step2_test3/model/student_SSKD_{epoch + 1}.pth')

    # 每个epoch打印一次训练时间
    end = time.perf_counter()
    runTime = end - start
    print(f"Epoch {epoch + 1}/{epochs}, 运行时间：{runTime}")

    # 每个epoch结束时计算平均损失
    epoch_loss = sum(loss_list) / len(loss_list)
    epoch_loss_list.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, pose平均损失: {epoch_loss}")

    with open('./outputs/SSKD/3/S_step2_test3/epoch_loss_list.txt', 'a') as file:
        file.write("%s\n" % epoch_loss)

    total_error = 0.0
    for a, data in enumerate(test_loader):
        images_nor = data["clr"].to(device)
        images_pro = data["clr_pro"].to(device)
        pose_gts_nor = data["joint_nor"].to(device)
        pose_gts_pro = data["joint_pro"].to(device)
        pose_gts = data["joint"].to(device)

        checkpoint_eval_student = torch.load(f'./outputs/SSKD/3/S_step2_test3/model/student_SSKD_{epoch + 1}.pth')
        student_eval.load_state_dict(checkpoint_eval_student['model_state_dict'])
        with torch.no_grad():
            student_eval.eval()
            out_S, z_S, feats_S = student_eval(images_nor, bb_grad=True)
            student_preds = out_S * (
                        torch.tensor(max_values[0]).to(device) - torch.tensor(min_values[0]).to(device)) + torch.tensor(
                min_values[0]).to(device)

            gtj, pred_align = global_align((pose_gts).cpu().numpy(), (student_preds).cpu().numpy())

            batch_error = 0.0
            for c in range(images_nor.size(0)):
                dist = torch.tensor(pred_align[c]) - torch.tensor(gtj[c])  # K x 3
                batch_error += dist.pow(2).sum(-1).sqrt().mean()
            batch_error /= images_nor.size(0)  # 计算当前批次的平均误差
            # 更新总误差
            total_error += batch_error.item()

            print(f"Epoch {epoch + 1}/{epochs},batch {a}/{len(test_loader)}, pose平均损失: {batch_error.item() * 10.0}")

    avg_est_error = total_error / len(test_loader)
    #     results_pose_cam_xyz.append(avg_est_error*10.0)
    print(f"Epoch {epoch + 1}/{epochs}, 总平均损失: {avg_est_error * 10.0}")

    with open('./outputs/SSKD/3/S_step2_test3/avg_est_error_list.txt', 'a') as file:
        file.write("%s\n" % (avg_est_error * 10.0))

