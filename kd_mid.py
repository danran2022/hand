import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os.path as osp
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# 检查是否有可用的CUDA设备
if torch.cuda.is_available():
    device = torch.device("cuda")  # 选择第一个可用的GPU设备
else:
    device = torch.device("cpu")  # 如果没有可用的GPU设备，则选择CPU
device

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

# 创建模型实例
# model2 = MobileNetV3_small_pro2()

# 创建学生网络实例并进行前向传播
student_net = MobileNetV3_small()
# student_net_step2 = SqueezeNetStudent()
student_net_eval = MobileNetV3_small()

# optimizer_step1 = torch.optim.Adam(filter(lambda p: p.requires_grad, student_net_step1.parameters()), lr=0.001)
optimizer = torch.optim.Adam(student_net.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

mse_loss = torch.nn.MSELoss()
# alpha = 0.25
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

loaded_data = np.load('./data/STB/data/min_max.npz')
min_values = loaded_data['array1']
max_values = loaded_data['array2']

# checkpoint_eval = torch.load(f'./outputs/student_net_epoch_123.pth')
#
# student_net_eval.load_state_dict(checkpoint_eval['model_state_dict'])
# optimizer.load_state_dict(checkpoint_eval['optimizer_state_dict'])

# epoch = 123
alpha = 0.25

for epoch in range(60, epochs):
    start = time.perf_counter()
    student_net.train()
    #     model.eval()
    loss_list = []
    scheduler.step()
    # 保存模型
    min_loss_index = -1
    min_loss = float('inf')
    r_t = 1 - (epoch / epochs)

    for i, metas in enumerate(train_loader):
        images_nor = metas["clr"]
        pose_gts_nor = metas["joint_nor"]
        pose_gts = metas["joint"]

        images_nor = images_nor.to(device)
        pose_gts_nor = pose_gts_nor.to(device)
        pose_gts = pose_gts.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            T_preds = model(images_nor)
        pose_preds_t = func.to_numpy(T_preds['xyz'] * 100.0)
        pose_gts, pose_preds_T = global_align(pose_gts.cpu().numpy(), pose_preds_t)

        pose_preds_T = (torch.tensor(pose_preds_T) - torch.tensor(min_values[0])) / (
                    torch.tensor(max_values[0]) - torch.tensor(min_values[0]))
        pose_preds_T = pose_preds_T.to(device)

        feature_S, pose_preds_S = student_net(images_nor)

        loss_stu = mse_loss(pose_preds_S.to(torch.float32), pose_gts_nor.to(torch.float32))

        # if torch.sum((pose_preds_S - pose_gts_nor) ** 2) > torch.sum((pose_preds_T - pose_gts_nor) ** 2):
        #     # loss_kd_out = torch.sum((pose_preds_S.to(torch.float32) - pose_preds_T.to(torch.float32)) ** 2) / batch_size
        #     loss_kd_out = mse_loss(pose_preds_S.to(torch.float32),pose_preds_T.to(torch.float32))
        # else:
        #     loss_kd_out = 0

        # loss_kd_out = mse_loss(pose_preds_S.to(torch.float32),pose_preds_T.to(torch.float32))
        # loss_kd_out = kl_divergence(pose_preds_S.to(torch.float32), pose_preds_T.to(torch.float32))

        loss_kd_fea = mse_loss(feature_S.to(torch.float32), T_preds['features_mid'].to(torch.float32))

        # loss = alpha * loss_stu + (1 - alpha) * (loss_kd_out + loss_kd_fea)
        loss = alpha * loss_stu + (1 - alpha) * loss_kd_fea
        # loss = loss_stu + r_t*alpha*loss_kd_fea + r_t*beta*loss_kd_out
        # loss = loss_stu + alpha*loss_kd_fea + beta*loss_kd_out
        # loss = loss_stu  + 0.00005 * loss_kd_fea

        # 反向传播，执行优化器
        loss.backward()
        optimizer.step()
        #         break
        loss_list.append(loss.item())  # 将每个batch的损失值添加到列表中

        # if loss_kd_out != 0:
        # print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()},loss_stu: {loss_stu.item()},loss_kd_fea: {loss_kd_fea.item()},loss_kd_out: {loss_kd_out}")
        # else:
        # print(f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()},loss_stu: {loss_stu.item()},loss_kd_fea: {loss_kd_fea.item()},loss_kd_out: {loss_kd_out.item()}")

        print(
            f"Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)}, Loss: {loss.item()},loss_stu: {loss_stu.item()},loss_kd_fea: {loss_kd_fea.item()}")
    # 保存模型

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': student_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 可选: 其他你想保存的内容
    }
    # torch.save(checkpoint,
    # f'D:\\hand recognition\\Minimal-Hand-pytorch-main\\outputs\\MobileNetV3_small_kd_test_3loss\\model\\student_net_epoch_{epoch + 1}.pth')
    torch.save(checkpoint,
               f'./outputs/MobileNetV2_kd_test_loss6/model/student_net_epoch_{epoch + 1}.pth')

    # 每个epoch打印一次训练时间
    end = time.perf_counter()
    runTime = end - start
    print(f"Epoch {epoch + 1}/{epochs}, 运行时间：{runTime}")

    # 每个epoch结束时计算平均损失
    epoch_loss = sum(loss_list) / len(loss_list)
    epoch_loss_list.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{epochs}, pose平均损失: {epoch_loss}")
    with open('./outputs/MobileNetV2_kd_test_loss6/epoch_loss_list.txt', 'a') as file:
        file.write("%s\n" % epoch_loss)

    total_error = 0.0
    for a, data in enumerate(test_loader):
        #         images_nor, images, cam_params, bboxes, pose_roots, pose_scales, image_ids ,image_paths,pose_gts,pose_gts_pro,pose_gts_nor= data
        #         pose_gts,pose_gts_nor, pose_gts_pro = dataset_test.pose_gts_val(image_ids)
        images_nor = data["clr"]
        pose_gts_nor = data["joint"]
        pose_gts = data["joint"]

        images_nor = images_nor.to(torch.float32)
        images_nor = images_nor.to(device)
        pose_gts = pose_gts.to(device)
        pose_gts_nor = pose_gts_nor.to(device)

        #         student_net_eval.load_state_dict(torch.load(
        #             f'D:\\hand recognition\\2019cvpr\\hand-graph-cnn-master\\output\\EfficientNet_SS\\model\\student_net_epoch_{epoch + 1}.pth'))
        checkpoint_eval = torch.load(f'./outputs/MobileNetV2_kd_test_loss6/model/student_net_epoch_{epoch + 1}.pth')
        student_net_eval.load_state_dict(checkpoint_eval['model_state_dict'])
        with torch.no_grad():
            student_net_eval.eval()
            feature_S, pose_preds_S = student_net_eval(images_nor)
            student_preds = pose_preds_S * (
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

    with open('./outputs/MobileNetV2_kd_test_loss6/avg_est_error_list.txt', 'a') as file:
        file.write("%s\n" % (avg_est_error * 10.0))

