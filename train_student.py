import argparse
import torch
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import numpy as np
import time

# from torch.utils.tensorboard import SummaryWriter

import scipy.io as sio
import os.path as osp
import logging
import cv2
import numpy as np
import numpy.linalg as LA
import math

import torch
import torch.utils.data

import argparse
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

import time
import json

import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim
import torchvision.transforms as transforms


import torch
import torchvision.transforms as transforms
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

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

# 检查是否有可用的CUDA设备
if torch.cuda.is_available():
    device = torch.device("cuda")  # 选择第一个可用的GPU设备
else:
    device = torch.device("cpu")  # 如果没有可用的GPU设备，则选择CPU

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

# 加载 MobileNetV3 Small
mobilenet_v3_small_model = mobilenet_v3_small(pretrained=True)

# 修改 MobileNetV3 Small 作为学生网络
class MobileNetV3_small(nn.Module):
    def __init__(self, num_keypoints):
        super(MobileNetV3_small, self).__init__()
        # 获取 MobileNetV3 Small 的 features 部分
        self.base_model = mobilenet_v3_small_model.features
        # 添加适应您需求的全局平均池化层和最后的线性层
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.last_linear = nn.Linear(576, num_keypoints * 3)

    def forward(self, x):
        x = self.base_model(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.last_linear(x)
        x = x.view(x.size(0), 21, 3)  # 调整输出形状为 20x3
        return x

# 创建学生网络实例并进行前向传播
student_net = MobileNetV3_small()
# student_net_step2 = SqueezeNetStudent()
student_net_eval = MobileNetV3_small()

# optimizer_step1 = torch.optim.Adam(filter(lambda p: p.requires_grad, student_net_step1.parameters()), lr=0.001)
optimizer = torch.optim.Adam(student_net.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

mse_loss = torch.nn.MSELoss()
alpha = 0.25
epochs = 200

student_net.to(device)
# student_net_step2.to(device)
student_net_eval.to(device)

# 计算模型大小并转换为MB
total_params = sum(p.numel() for p in student_net.parameters())
total_size = total_params / (1024 ** 2)
print(f"模型的总参数量为：{total_params},为{total_size:.2f} MB")

# 在每个epoch循环外部初始化一个空列表来保存每个epoch的损失
epoch_loss_list = []
avg_est_error_list = []

for epoch in range(epochs):
    start = time.perf_counter()
    student_net.train()
    #     model.eval()
    loss_list = []

    # 保存模型
    min_loss_index = -1
    min_loss = float('inf')
    for i, metas in enumerate(train_loader):
        images_nor = metas["clr"]
        pose_gts_nor = metas["joint_nor"]

        images_nor = images_nor.to(device)
        pose_gts_nor = pose_gts_nor.to(device)

        optimizer.zero_grad()

        feature_S, student_preds = student_net(images_nor)
        loss = mse_loss(student_preds.to(torch.float32), pose_gts_nor.to(torch.float32))

        loss.backward()
        optimizer.step()
        #         scheduler.step()

        loss_list.append(loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()}")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': student_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 可选: 其他你想保存的内容
    }
    torch.save(checkpoint,
               f'D:\\hand recognition\\Minimal-Hand-pytorch-main\\output\\resnet8\\model\\student_net_epoch_{epoch + 1}.pth')
    # 每个epoch打印一次训练时间
    end = time.perf_counter()
    runTime = end - start
    print(f"Epoch {epoch + 1}/{epochs}, 运行时间：{runTime}")

    # 每个epoch结束时计算平均损失
    epoch_loss = sum(loss_list) / len(loss_list)
    epoch_loss_list.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, pose平均损失: {epoch_loss}")
    with open('D:\\hand recognition\\Minimal-Hand-pytorch-main\\output\\resnet8\\epoch_loss_list.txt', 'a') as file:
        file.write("%s\n" % epoch_loss)

    total_error = 0.0
    results_pose_cam_xyz = {}
    # 每个epoch计算一次平均姿态误差
    for a, data in enumerate(test_loader):
        #         images_nor, images, cam_params, bboxes, pose_roots, pose_scales, image_ids ,image_paths,pose_gts,pose_gts_pro,pose_gts_nor= data
        #         pose_gts,pose_gts_nor, pose_gts_pro = dataset_test.pose_gts_val(image_ids)
        images_nor = data["clr"]
        pose_gts_nor = data["joint_nor"]
        pose_gts = data["joint"]

        images_nor = images_nor.to(torch.float32)
        images_nor = images_nor.to(device)
        pose_gts = pose_gts.to(device)
        pose_gts_nor = pose_gts_nor.to(device)

        #         student_net_eval.load_state_dict(torch.load(
        #             f'D:\\hand recognition\\2019cvpr\\hand-graph-cnn-master\\output\\EfficientNet_SS\\model\\student_net_epoch_{epoch + 1}.pth'))
        checkpoint_eval = torch.load(
            f'D:\\hand recognition\\Minimal-Hand-pytorch-main\\output\\resnet8\\model\\student_net_epoch_{epoch + 1}.pth')
        student_net_eval.load_state_dict(checkpoint_eval['model_state_dict'])
        with torch.no_grad():
            student_net_eval.eval()
            feature_S, student_preds = student_net(images_nor)

            loaded_data = np.load('./dataset/STB/data/min_max.npz')
            min_values = loaded_data['array1']
            max_values = loaded_data['array2']
            student_preds = student_preds * (
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

    with open('D:\\hand recognition\\Minimal-Hand-pytorch-main\\output\\resnet8\\avg_est_error_list.txt', 'a') as file:
        file.write("%s\n" % (avg_est_error * 10.0))