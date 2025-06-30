# -*- coding: utf-8 -*-
"""
validate the trained MedSAM model
"""

import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from tqdm import tqdm
from segment_anything import sam_model_registry
import argparse
import random
import glob
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.ndimage.morphology import generate_binary_structure
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from monai.losses import DiceLoss

# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

def save_fusion_image(original_img, mask, save_path, mask_color=[255, 0, 0], alpha=0.5):
    # 确保 mask 是二维的 (H, W)，如果是 (1, H, W) 则进行 squeeze
    if len(mask.shape) == 3:
        mask = mask.squeeze(0)

    # 创建融合图像
    fused_img = original_img.copy()

    # 将掩码的区域设为指定的颜色
    colored_mask = np.zeros_like(fused_img)  # 创建与原始图像相同大小的空白图像
    colored_mask[mask > 0] = mask_color  # 只给掩码区域着色

    # 融合图像和掩码，alpha 表示掩码的透明度
    fused_img = np.uint8(fused_img * (1 - alpha) + colored_mask * alpha)

    # 保存融合后的图像
    plt.imsave(save_path, fused_img)

def calculate_iou(pred_mask, gt_mask, threshold=0):
    # 1. 二值化预测掩码，使用阈值将预测值大于阈值的部分视为前景
    pred_mask = (pred_mask > threshold).astype(bool)
    
    # 2. 将真实掩码转换为布尔值类型
    gt_mask = gt_mask.astype(bool)
    print(pred_mask.shape, gt_mask.shape)

    # 3. 计算交集和并集
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    # 4. 计算 IOU，如果并集为 0，返回 NaN 避免除零错误
    if union == 0:
        return 0.0
    else:
        return intersection / union


def calculate_dice_score(pred_mask, gt_mask, threshold=0):
    # 1. 二值化预测掩码
    pred_mask = (pred_mask > threshold).astype(bool)
    
    # 2. 将真实掩码转换为布尔值
    gt_mask = gt_mask.astype(bool)

    # 3. 计算交集和总像素数
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pixels = pred_mask.sum() + gt_mask.sum()

    # 4. 计算 Dice Score，如果像素总数为 0，返回 NaN 避免除零错误
    if total_pixels == 0:
        return 0.0
    else:
        return (2. * intersection) / total_pixels



def calculate_hd95(pred_mask, gt_mask, threshold=0):
    """
    计算预测掩码与真实掩码之间的 95th percentile Hausdorff Distance (HD95)。

    Args:
        pred_mask (np.ndarray): 模型输出的预测概率图或二值掩码，形状 (H, W) 或 (1, H, W)。
        gt_mask   (np.ndarray): 真实二值掩码，形状 (H, W) 或 (1, H, W)。
        threshold (float):     将预测概率图二值化的阈值，默认 0.5。

    Returns:
        float: 预测和真值之间的 HD95，单位：像素。
    """
    # 1. 二值化
    pred = (pred_mask > threshold).squeeze().astype(bool)
    gt   = (gt_mask   > threshold).squeeze().astype(bool)

    # 2. 提取边界——原 mask 异或 腐蚀后的 mask
    footprint   = generate_binary_structure(pred.ndim, 1)
    pred_border = pred ^ binary_erosion(pred, structure=footprint, iterations=1)
    gt_border   = gt   ^ binary_erosion(gt,   structure=footprint, iterations=1)

    # 3. 计算“预测边界到真值边界”的距离
    #    distance_transform_edt(~gt_border) 生成一个场，表示每个像素到最近 gt_border 的距离
    dt_gt = distance_transform_edt(~gt_border)
    d_pred_to_gt = dt_gt[pred_border]
    # 取 95th percentile，如果没有边界点则默认为 0
    hd_pred_to_gt = np.percentile(d_pred_to_gt, 95) if d_pred_to_gt.size > 0 else 0.0

    # 4. 计算“真值边界到预测边界”的距离
    dt_pred = distance_transform_edt(~pred_border)
    d_gt_to_pred = dt_pred[gt_border]
    hd_gt_to_pred = np.percentile(d_gt_to_pred, 95) if d_gt_to_pred.size > 0 else 0.0

    # 5. 返回两个方向距离的最大值
    return max(hd_pred_to_gt, hd_gt_to_pred)

def calculate_asd(pred_mask, gt_mask, threshold=0):
    # 二值化
    pred = (pred_mask > threshold)
    gt   = (gt_mask  > threshold)

    # 取边界：原 mask 异或 腐蚀后的 mask
    footprint   = generate_binary_structure(pred.ndim, 1)
    pred_border = pred ^ binary_erosion(pred, structure=footprint, iterations=1)
    gt_border   = gt   ^ binary_erosion(gt,   structure=footprint, iterations=1)

    # 真值边界到预测边界的距离
    # distance_transform_edt 的输入是“非边界”为 True，边界点为 False
    dt_pred = distance_transform_edt(~pred_border)
    dists_gt_to_pred = dt_pred[gt_border]

    # 预测边界到真值边界的距离
    dt_gt = distance_transform_edt(~gt_border)
    dists_pred_to_gt = dt_gt[pred_border]

    # 合并两组距离
    all_dists = np.concatenate([
        dists_gt_to_pred,
        dists_pred_to_gt
    ]) if (dists_gt_to_pred.size>0 and dists_pred_to_gt.size>0) else \
        (dists_gt_to_pred if dists_pred_to_gt.size==0 else dists_pred_to_gt)

    # 防止空数组
    if all_dists.size == 0:
        return 0.0
    return all_dists.mean()

def calculate_metrics(pred_mask, gt_mask, threshold=0):
    # 二值化预测掩码
    pred_mask = (pred_mask > threshold).astype(bool)
    
    # 将真实掩码转换为布尔值类型
    gt_mask = gt_mask.astype(bool)

    # 计算 TP, TN, FP, FN
    TP = np.logical_and(pred_mask, gt_mask).sum()  # True Positives
    TN = np.logical_and(~pred_mask, ~gt_mask).sum()  # True Negatives
    FP = np.logical_and(pred_mask, ~gt_mask).sum()  # False Positives
    FN = np.logical_and(~pred_mask, gt_mask).sum()  # False Negatives

    # 计算 Accuracy (Acc)
    total = TP + TN + FP + FN
    Acc = (TP + TN) / total if total > 0 else 0.0

    # 计算 Specificity (Spe)
    Spe = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # 计算 Sensitivity (Sen)
    Sen = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # 计算 Jaccard Index (Jac)
    Jac = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    return Acc, Spe, Sen, Jac

# 初始化 DiceLoss
seg_loss = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")

# Import your dataset class
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
    # def __init__(self, data_root, bbox_shift=2):# for eyes
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.gt_path_files[index]
        )
        label_ids = np.unique(gt)[1:]
        masks = []
        bboxes = []
        label_list = []

        for label_id in label_ids:
            # 生成该标签的二值化掩码
            gt2D = np.uint8(gt == label_id)
            y_indices, x_indices = np.where(gt2D > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue  # 如果没有有效的掩码，跳过
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, self.bbox_shift))
            x_max = min(W, x_max + random.randint(0, self.bbox_shift))
            y_min = max(0, y_min - random.randint(0, self.bbox_shift))
            y_max = min(H, y_max + random.randint(0, self.bbox_shift))
            bbox = np.array([x_min, y_min, x_max, y_max])

            # 保存该标签的掩码和边界框
            masks.append(torch.tensor(gt2D[None, :, :], dtype=torch.uint8))  # 添加一个 channel 维度
            bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            label_list.append(label_id)  # 保存标签 ID

        return (
            torch.tensor(img_1024).float(),  # 返回图像
            masks,  # 返回所有标签的掩码列表
            bboxes,  # 返回所有标签的边界框列表
            label_list,  # 返回所有标签的 ID 列表
            img_name,  # 返回图像名称
        )

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--data_path",
    type=str,
    # default="data_ex3_amos/test/npy/amos_2022",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument(
    "-o",
    "--seg_path",
    type=str,
    # default="data_ex3_amos/res/data/adapt+de-amos-02241010",
    help="path to the segmentation folder",
)

parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-chk",
    "--checkpoint",
    type=str,
    # default="work_dir/MedSAM/medsam_vit_b.pth",
    help="path to the trained model",
)
# parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=10)

parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()

device = torch.device(args.device)

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        fat_net,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        """增加FAT_Net网络中cnn模块"""
        self.fat_net = fat_net

        # 冻结 SAM 的其他部分
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            # param.requires_grad = True
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        # 允许更新 `fat_net` 的参数
        for param in self.fat_net.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        # 1. 提取 SAM 编码器特征
        image_embedding_sam = self.image_encoder(image)  # (B, 256, 64, 64)
        # 2. 使用 FAT_Net 进行特征融合，传入 SAM 编码器特征和 CNN 特征
        fused_embedding, cnn_features = self.fat_net(image_embedding_sam, image)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )

        low_res_masks, _ = self.mask_decoder(
            image_embeddings=fused_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            cnn_features=cnn_features,  # 传递多尺度特征
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

class AttentionFAM(nn.Module):
    def __init__(self, initial_bias=0.8):
        super(AttentionFAM, self).__init__()

        # 计算注意力权重的 1x1 卷积层
        self.attention_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # 将偏向因子变为一个可学习参数
        self.bias_factor = nn.Parameter(torch.tensor(initial_bias))

    def forward(self, feature_sam, feature_cnn):
        # 将 CNN 特征上采样至与 SAM 特征相同的大小
        feature_cnn = F.interpolate(feature_cnn, size=feature_sam.shape[-2:], mode="bilinear", align_corners=False)
        
        # 计算注意力权重
        combined_features = torch.cat([feature_sam, feature_cnn], dim=1)
        attention_weights = self.sigmoid(self.attention_conv(combined_features))

        # 使用可学习的偏向因子调整权重
        attention_weights = attention_weights * (1 - self.bias_factor) + self.bias_factor

        # 融合特征
        fused_features = attention_weights * feature_sam + (1 - attention_weights) * feature_cnn
        return fused_features

# FAT_Net 模块
class FAT_Net(nn.Module):
    def __init__(self):
        super(FAT_Net, self).__init__()
        self.cnn = CNN_Module()  # 定义 CNN 模块
        self.fam = AttentionFAM()  # 定义 FAM 模块

    def forward(self, feature_sam, image):
        # 使用 CNN 提取多尺度特征
        scale1, scale2, scale3, scale4 = self.cnn(image)

        # 使用 AttentionFAM 融合最高层的 CNN 特征和 SAM 特征
        fused_features = self.fam(feature_sam, scale3)  # 仅融合最高层特征

        # 返回融合后的最高层特征和所有多尺度特征
        return fused_features, [scale1, scale2, scale3, scale4]

# CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_weights

        return x


# 残差块模块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入通道和输出通道不一致，则添加一个额外的卷积层来调整维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 是否引入 CBAM 模块
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)  # Shortcut (直接连接或调整通道)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # 加上输入的 shortcut 分支
        if self.use_cbam:
            out = self.cbam(out)  # 引入 CBAM
        out = self.relu(out)
        return out


# CNN 模块
class CNN_Module(nn.Module):
    def __init__(self):
        super(CNN_Module, self).__init__()
        self.block1 = ResidualBlock(3, 64, use_cbam=True)  # 添加 CBAM
        self.block2 = ResidualBlock(64, 128, stride=2, use_cbam=True)
        self.block3 = ResidualBlock(128, 256, stride=2, use_cbam=True)
        self.block4 = ResidualBlock(256, 512, stride=2, use_cbam=True)

    def forward(self, x):
        # 提取多尺度特征
        scale1 = self.block1(x)  # (B, 64, H, W)
        scale2 = self.block2(scale1)  # (B, 128, H/2, W/2)
        scale3 = self.block3(scale2)  # (B, 256, H/4, W/4)
        scale4 = self.block4(scale3)  # (B, 512, H/8, W/8)

        return scale1, scale2, scale3, scale4


def main():
    # Load model
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    fat_net = FAT_Net()  # 初始化 FAT_Net

    if args.checkpoint is not None:
        with open(args.checkpoint, "rb") as f:
            checkpoint_dict = torch.load(f)

        # 提取 fat_net 相关的权重
        fat_net_state_dict = {k: v for k, v in checkpoint_dict.items() if "fat_net" in k}

        # 创建一个新的字典，去掉 "fat_net." 前缀
        new_fat_net_state_dict = {}
        for key, value in fat_net_state_dict.items():
            new_key = key.replace("fat_net.", "")  # 去掉 "fat_net." 前缀
            new_fat_net_state_dict[new_key] = value

        # 加载新的权重字典
        fat_net.load_state_dict(new_fat_net_state_dict, strict=False)

        print("Fat_net weights loaded successfully!")

    # 打印fat_net的参数
    for name, param in fat_net.named_parameters():
        print(f"{name}: {param.shape}")

    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        fat_net=fat_net,
    ).to(device)

    # 设置模型为评估模式，确保冻结参数
    medsam_model.eval()

    # Load validation dataset
    val_dataset = NpyDataset(args.data_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 初始化用于存储评估指标的列表
    iou_values = []
    dice_scores = []
    hd95_values = []
    asd_values = []

    # 新增四个评价指标的列表
    acc_values = []
    spe_values = []
    sen_values = []
    jac_values = []
    dice_loss_values = []

    # 创建保存目录
    os.makedirs(args.seg_path, exist_ok=True)
    os.makedirs(join(args.seg_path, "original"), exist_ok=True)  # 保存原图的文件夹
    os.makedirs(join(args.seg_path, "predicted_mask"), exist_ok=True)  # 保存预测 mask 的文件夹
    os.makedirs(join(args.seg_path, "ground_truth_mask"), exist_ok=True)  # 保存真实 mask 的文件夹
    os.makedirs(join(args.seg_path, "fusion"), exist_ok=True)  # 保存真实 mask 的文件夹
    # 创建颜色掩码
    color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])  # 默认颜色


    # CSV 文件路径
    csv_file_path = join(args.seg_path, 'results.csv')

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Name', 'label', 'IOU', 'DICE', 'HD95', 'ASD', 'Accuracy', 'Specificity', 'Sensitivity', 'Jaccard Index', 'Dice Loss', 'Dice Loss'])

    with torch.no_grad():
        for step, (image, masks, boxes, label_list, names) in enumerate(tqdm(val_dataloader)):
            image = image.to(device)
            boxes = [box.to(device) for box in boxes]  # 将每个 box 转移到 GPU

            # 保存图像和 mask
            for i in range(image.size(0)):  # 处理每个 batch 中的图像
                img_name = names[i].split('.')[0]  # 获取图像的名称（不包括扩展名）

                # 保存原图 (原图数值可能在 [0, 1] 范围，需要乘以 255)
                original_img = image[i].cpu().permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
                original_img = (original_img * 255).astype(np.uint8)
                plt.imsave(join(args.seg_path, "original", f"{img_name}_1-1.jpg"), original_img)

                # 对每个标签单独处理
                for j, (mask, bbox, label_id) in enumerate(zip(masks, boxes, label_list)):
                    mask = mask.to(device)
                    bbox_np = bbox.detach().cpu().numpy()

                    # 获取预测掩码
                    medsam_pred = medsam_model(image[i:i+1], bbox_np)
                    pred_mask = medsam_pred.squeeze().cpu().numpy()
                    # 计算掩码图像的大小
                    h, w = pred_mask.shape[-2:]
                    mask_image = pred_mask.reshape(h, w, 1) * color.reshape(1, 1, -1) 
                    # 将 mask_image 值限制在 [0, 1] 范围内
                    mask_image = np.clip(mask_image, 0, 1) 

                    # 计算当前标签的 IOU 和 Dice
                    gt_mask = mask.squeeze().cpu().numpy()

                    # 保存预测掩码和评估结果
                    plt.imsave(join(args.seg_path, "predicted_mask", f"{img_name}_label{label_id}_pred.jpg"), mask_image, cmap='gray')
                    plt.imsave(join(args.seg_path, "ground_truth_mask", f"{img_name}_label{label_id}_gt.jpg"), gt_mask, cmap='gray')
                    fusion_save_path = join(args.seg_path, "fusion", f"{img_name}_label{label_id}_fusion.jpg")
                    save_fusion_image(original_img, pred_mask, fusion_save_path, mask_color=[255, 0, 0], alpha=0.5)

                    # Skip if masks are empty
                    if np.all(pred_mask == 0) or np.all(gt_mask == 0):
                        print(f"Empty mask for label {label_id}, skipping HD95 and ASD calculation")
                        continue

                    iou = calculate_iou(pred_mask, gt_mask)
                    dice_score = calculate_dice_score(pred_mask, gt_mask)
                    # 计算 HD95 和 ASD
                    hd95 = calculate_hd95(pred_mask, gt_mask)
                    asd = calculate_asd(pred_mask, gt_mask)

                    # 打印和记录结果
                    acc, spe, sen, jac = calculate_metrics(pred_mask, gt_mask)

                    # Calculate Dice Loss using MONAI's DiceLoss
                    pred_mask_tensor = torch.tensor(pred_mask).unsqueeze(0).unsqueeze(0).float().to(device)
                    gt_mask_tensor = torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0).float().to(device)
                    dice_loss = seg_loss(pred_mask_tensor, gt_mask_tensor).item()  # .item() gets the scalar value

                    print(f"IOU for label {label_id}: {iou}, Dice Score: {dice_score}, HD95: {hd95}, ASD: {asd}, Accuracy: {acc}, Specificity: {spe}, Sensitivity: {sen}, Jaccard Index: {jac}")


                    # 将结果保存到 CSV 文件
                    with open(csv_file_path, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f"{img_name}_label{label_id}", label_id, iou, dice_score, hd95, asd, acc, spe, sen, jac, dice_loss])

                    # 记录 IOU 和 Dice Score
                    iou_values.append(iou)
                    dice_scores.append(dice_score)
                    hd95_values.append(hd95)
                    asd_values.append(asd)

                    # 新增：将四个新指标的值添加到各自的列表
                    acc_values.append(acc)
                    spe_values.append(spe)
                    sen_values.append(sen)
                    jac_values.append(jac)
                    dice_loss_values.append(dice_loss)


    print('val end')


if __name__ == "__main__":
    main()