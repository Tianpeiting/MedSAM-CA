# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob
import logging
import time


def set_random_seed(seed):
    """Seed all RNGs used by sampling, box perturbation, and training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


torch.cuda.empty_cache()


os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def calculate_iou(pred_mask, gt_mask, threshold=0):
    # 1. 二值化预测掩码，使用阈值将预测值大于阈值的部分视为前景
    pred_mask = (pred_mask > threshold).to(torch.bool)  # 修正这里

    # 2. 将真实掩码转换为布尔值类型
    gt_mask = gt_mask.to(torch.bool)  # 修正这里

    # 3. 计算交集和并集
    intersection = (pred_mask & gt_mask).sum()  # 使用&来表示逻辑与操作
    union = (pred_mask | gt_mask).sum()  # 使用|来表示逻辑或操作

    # 4. 计算 IOU，如果并集为 0，返回 0 避免除零错误
    if union == 0:
        return 0.0
    else:
        return 1 - intersection / union

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

# 边界感知损失的实现
def edge_loss(pred, gt):
    # 使用Sobel算子计算边缘（x方向和y方向）
    sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)

    # 用于提取预测和标签的边缘
    pred_edge_x = F.conv2d(pred, sobel_kernel_x, padding=1)
    pred_edge_y = F.conv2d(pred, sobel_kernel_y, padding=1)
    gt_edge_x = F.conv2d(gt, sobel_kernel_x, padding=1)
    gt_edge_y = F.conv2d(gt, sobel_kernel_y, padding=1)

    # 计算边缘差异
    edge_diff_x = torch.abs(pred_edge_x - gt_edge_x)
    edge_diff_y = torch.abs(pred_edge_y - gt_edge_y)

    # 汇总x方向和y方向的边缘差异作为边缘损失
    edge_loss_value = torch.mean(edge_diff_x + edge_diff_y)

    return edge_loss_value


class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20, augment=False):
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
        self.augment = augment
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        gt = np.load(
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (
            "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        if self.augment:
            if random.random() < 0.5:
                img_1024 = np.flip(img_1024, axis=1)
                gt = np.flip(gt, axis=1)
            if random.random() < 0.5:
                img_1024 = np.flip(img_1024, axis=0)
                gt = np.flip(gt, axis=0)
            rotation_k = random.randint(0, 3)
            if rotation_k:
                img_1024 = np.rot90(img_1024, rotation_k, axes=(0, 1))
                gt = np.rot90(gt, rotation_k, axes=(0, 1))
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(-0.05, 0.05)
            img_1024 = np.clip(
                (img_1024 - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0
            )

        # Copy removes negative strides introduced by flips/rotations.
        img_1024 = np.transpose(img_1024, (2, 0, 1)).copy()
        gt = np.asarray(gt).copy()
        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            img_name,
        )

# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    # Keep the 20-case subset identical to the seed-2025 (0743) run so that
    # this experiment changes only the training RNG seed.
    default="../../../data/tpt_data/data_ex2_mel/data_ex2_mel/2016+2017+2018/train0.4_2/npy/ISIC_mel",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("-task_name", type=str, default="MedSAM-ViT-B")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument(
    "-checkpoint", type=str, default="/data/tpt_data/work_dir/adapt/medsam_vit_b.pth"
)
parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)
parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="./work_dir/suijishuxiaorong/")
# train
parser.add_argument("-num_epochs", type=int, default=80)
parser.add_argument("-batch_size", type=int, default=8)
parser.add_argument("-num_workers", type=int, default=0)
parser.add_argument(
    "--seed",
    type=int,
    default=2021,
    help="Random seed for subset-run reproducibility.",
)
parser.add_argument(
    "--reinit_train_adapters",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "By default, reinitialize all 12 image-encoder Space_Adapter modules from the run "
        "seed and include only those adapter parameters in optimization; the "
        "remaining ViT image encoder stays frozen."
    ),
)
parser.add_argument(
    "--bbox_shift",
    type=int,
    default=20,
    help="Maximum random training/validation box perturbation in pixels.",
)
parser.add_argument(
    "--val_npy_path",
    type=str,
    default="../../../data/tpt_data/data_ex2_mel/data_ex2_mel/2016+2017+2018/val/npy/ISIC_mel",
    help="Path to validation npy files.",
)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    # Default training configuration follows the manuscript.
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "--lr_schedule",
    choices=("constant", "cosine"),
    default="constant",
    help="Epoch-level learning-rate schedule.",
)
parser.add_argument(
    "--min_lr",
    type=float,
    default=0.00001,
    help="Minimum learning rate for cosine annealing.",
)
parser.add_argument(
    "--train_augment",
    action="store_true",
    help="Apply paired flips/90-degree rotations and mild color jitter to training data.",
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=False, help="use amp")
parser.add_argument(
    "--resume", type=str, default="", help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
set_random_seed(args.seed)

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)
# %% set up model


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
        for param in self.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False


        # 允许更新 `mask_decoder` 层
        for name, param in self.mask_decoder.named_parameters():
            if "attention_fusion" in name:
                param.requires_grad = True
            if "output_upscaling" in name:
                param.requires_grad = True

        # 允许更新 `fat_net` 的参数
        for param in self.fat_net.parameters():
            param.requires_grad = True

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


def reinitialize_and_unfreeze_space_adapters(image_encoder):
    """Reset the 12 bottleneck adapters while leaving the ViT backbone frozen."""
    adapter_modules = []
    for block_index, block in enumerate(image_encoder.blocks):
        adapter = getattr(block, "Space_Adapter", None)
        if adapter is None:
            raise RuntimeError(
                f"Image-encoder block {block_index} has no Space_Adapter module."
            )
        # Adapter itself has no reset_parameters method; its two Linear layers do.
        for child in adapter.modules():
            if child is not adapter and hasattr(child, "reset_parameters"):
                child.reset_parameters()
        for parameter in adapter.parameters():
            parameter.requires_grad = True
        adapter_modules.append(adapter)

    if len(adapter_modules) != 12:
        raise RuntimeError(
            f"Expected 12 Space_Adapter modules, found {len(adapter_modules)}."
        )
    parameter_count = sum(
        parameter.numel()
        for adapter in adapter_modules
        for parameter in adapter.parameters()
    )
    return len(adapter_modules), parameter_count

def load_compatible_optimizer(optimizer, saved_optimizer, model_state, named_parameters,
                              renamed=None, removed=()):
    """Migrate the legacy single-group optimizer using model parameter order."""
    renamed = renamed or {}
    named_parameters = list(named_parameters)
    current_names = [name for name, _ in named_parameters]
    current_group = optimizer.state_dict()["param_groups"]
    saved_groups = saved_optimizer["param_groups"]
    if len(saved_groups) != 1 or len(current_group) != 1:
        raise ValueError("Legacy optimizer migration requires one parameter group")
    if [id(p) for _, p in named_parameters] != [
        id(p) for p in optimizer.param_groups[0]["params"]
    ]:
        raise ValueError("Optimizer parameter order does not match named_parameters")

    # state_dict follows the same module/parameter order as named_parameters.
    accepted = set(current_names) | set(renamed) | set(removed)
    old_names = [name for name in model_state if name in accepted]
    old_ids = saved_groups[0]["params"]
    mapped_names = [renamed.get(name, name) for name in old_names if name not in removed]
    if len(old_names) != len(old_ids) or mapped_names != current_names:
        raise ValueError(
            "Legacy optimizer layout differs beyond the bias changes. "
            "Use the same Adapter training setting as the saved run."
        )
    old_id_by_name = dict(zip(old_names, old_ids))
    original_name = {new: old for old, new in renamed.items()}
    migrated_state = {}
    new_ids = current_group[0]["params"]
    for (name, parameter), new_id in zip(named_parameters, new_ids):
        old_name = original_name.get(name, name)
        old_id = old_id_by_name[old_name]
        if old_id not in saved_optimizer["state"]:
            continue
        state = saved_optimizer["state"][old_id]
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in state and state[key].shape != parameter.shape:
                raise ValueError(f"Optimizer state shape mismatch for {name}: {key}")
        migrated_state[new_id] = state
    group = dict(saved_groups[0])
    group["params"] = new_ids
    if "param_names" in group:
        group["param_names"] = current_names
    optimizer.load_state_dict({"state": migrated_state, "param_groups": [group]})


class AttentionFAM(nn.Module):
    def __init__(self, initial_bias=0.8):
        super(AttentionFAM, self).__init__()

        # 计算注意力权重的 1x1 卷积层
        self.attention_conv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Optimize beta while constraining the effective SAM lower bound to (0, 1).
        if not 0.0 < initial_bias < 1.0:
            raise ValueError("initial_bias must lie strictly between 0 and 1")
        self.beta = nn.Parameter(torch.logit(torch.tensor(initial_bias, dtype=torch.float32)))

    @property
    def bias_factor(self):
        """Effective bias b = sigmoid(beta), retained as a readable attribute."""
        return torch.sigmoid(self.beta)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Old checkpoints store b directly. Convert it without changing inference.
        legacy_key, beta_key = prefix + "bias_factor", prefix + "beta"
        if legacy_key in state_dict:
            legacy_bias = state_dict.pop(legacy_key)
            if beta_key in state_dict:
                error_msgs.append(f"Both {legacy_key} and {beta_key} are present")
            elif legacy_bias.numel() != 1 or not bool(
                torch.isfinite(legacy_bias).all()
                & (legacy_bias > 0).all() & (legacy_bias < 1).all()
            ):
                error_msgs.append(
                    f"Cannot convert {legacy_key}: the old bias must be finite and strictly between 0 and 1"
                )
            else:
                state_dict[beta_key] = torch.logit(legacy_bias.reshape(()))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, feature_sam, feature_cnn):
        # 将 CNN 特征上采样至与 SAM 特征相同的大小
        feature_cnn = F.interpolate(feature_cnn, size=feature_sam.shape[-2:], mode="bilinear", align_corners=False)
        
        # 计算注意力权重
        combined_features = torch.cat([feature_sam, feature_cnn], dim=1)
        attention_weights = self.sigmoid(self.attention_conv(combined_features))

        # 使用可学习的偏向因子调整权重
        bias = self.bias_factor
        attention_weights = attention_weights * (1 - bias) + bias

        # 融合特征
        fused_features = attention_weights * feature_sam + (1 - attention_weights) * feature_cnn
        return fused_features
    

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
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )


    # 创建日志文件
    log_file_path = join(model_save_path, run_id + "_training_log.txt")
    logging.basicConfig(
        filename=log_file_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # 同时输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info("Starting Training...")
    logging.info(f"Random seed: {args.seed}")
    logging.info(f"Initialization checkpoint: {args.checkpoint}")
    logging.info(f"Training data: {args.tr_npy_path}")
    logging.info(f"Validation data: {args.val_npy_path}")
    logging.info(f"BBox shift: {args.bbox_shift}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"LR schedule: {args.lr_schedule}; minimum LR: {args.min_lr}")
    logging.info(f"Training augmentation: {args.train_augment}")

    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    # 创建 FAT_Net 实例
    fat_net = FAT_Net()

    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
        fat_net=fat_net,  # 传入 FAT_Net 实例
    ).to(device)
    if args.reinit_train_adapters:
        adapter_module_count, adapter_parameter_count = (
            reinitialize_and_unfreeze_space_adapters(medsam_model.image_encoder)
        )
        logging.info(
            "Adapter protocol: reinitialized %d Space_Adapter modules (%d "
            "trainable parameters) using random seed %d.",
            adapter_module_count,
            adapter_parameter_count,
            args.seed,
        )
    else:
        logging.info("Adapter protocol: checkpoint adapters remain frozen.")
    medsam_model.train()



    logging.info(
        f"Number of total parameters: {sum(p.numel() for p in medsam_model.parameters())}"
    )
    logging.info(
        f"Number of trainable parameters: {sum(p.numel() for p in medsam_model.parameters() if p.requires_grad)}"
    )

    fat_net_params = list(medsam_model.fat_net.parameters())

    # Only reinitialized adapters can be trainable in the otherwise frozen encoder.
    img_mask_enc_params = [
        param for param in medsam_model.image_encoder.parameters()
        if param.requires_grad
    ]

    # 过滤出所有需要更新的参数
    img_mask_dec_params = [param for param in medsam_model.mask_decoder.parameters() if param.requires_grad]

    img_mask_encdec_params = img_mask_enc_params + img_mask_dec_params

    # 创建优化器，只包含需要更新的参数
    optimizer_sam = torch.optim.AdamW(
        img_mask_encdec_params,  # 只传递 requires_grad=True 的参数
        lr=args.lr,  # 较小的学习率
        weight_decay=args.weight_decay
    )

    optimizer_fatnet = torch.optim.AdamW(
        list(fat_net_params),
        lr=args.lr,  # 正常的学习率
        weight_decay=args.weight_decay
    )

    logging.info(
        f"Number of image encoder and mask decoder parameters: {sum(p.numel() for p in fat_net_params if p.requires_grad)}"
    )


    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    losses = []
    val_losses = []
    best_loss = 1e10
    train_dataset = NpyDataset(
        args.tr_npy_path,
        bbox_shift=args.bbox_shift,
        augment=args.train_augment,
    )

    logging.info(f"Number of training samples: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 定义验证数据集
    val_dataset = NpyDataset(args.val_npy_path, bbox_shift=args.bbox_shift)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    def validate(model, dataloader, seg_loss, ce_loss):
        model.eval()  # 设置模型为评估模式
        val_loss = 0
        with torch.no_grad():  # 禁用梯度计算
            for step, (image, gt2D, boxes, _) in enumerate(tqdm(dataloader)):
                boxes_np = boxes.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)
                medsam_pred = model(image, boxes_np)
                loss = 0.5 * seg_loss(medsam_pred, gt2D) + 0.5 * ce_loss(medsam_pred, gt2D.float())
                val_loss += loss.item()
        model.train()  # 恢复模型为训练模式
        return val_loss / len(dataloader)

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            model_state = checkpoint["model"]
            legacy_bias_format = "fat_net.fam.bias_factor" in model_state or any(
                key.startswith("mask_decoder.attention_fusion.")
                and key.endswith(".bias_factor") for key in model_state
            )
            medsam_model.load_state_dict(model_state)
            if legacy_bias_format:
                sam_named = [
                    (name, parameter) for name, parameter in medsam_model.named_parameters()
                    if parameter.requires_grad and name.startswith(("image_encoder.", "mask_decoder."))
                ]
                removed_biases = [
                    key for key in model_state
                    if key.startswith("mask_decoder.attention_fusion.")
                    and key.endswith(".bias_factor")
                ]
                load_compatible_optimizer(
                    optimizer_sam, checkpoint["optimizer_sam"], model_state,
                    sam_named, removed=removed_biases,
                )
                fat_named = [
                    ("fat_net." + name, parameter)
                    for name, parameter in medsam_model.fat_net.named_parameters()
                ]
                renamed_bias = (
                    {"fat_net.fam.bias_factor": "fat_net.fam.beta"}
                    if "fat_net.fam.bias_factor" in model_state else {}
                )
                load_compatible_optimizer(
                    optimizer_fatnet, checkpoint["optimizer_fatnet"], model_state,
                    fat_named, renamed=renamed_bias,

                )
                logging.info(
                    "Loaded legacy weights and retained compatible optimizer states. "
                    "Unused decoder biases were removed; the converted beta retains the saved optimizer state."
                )
            else:
                optimizer_sam.load_state_dict(checkpoint["optimizer_sam"])
                optimizer_fatnet.load_state_dict(checkpoint["optimizer_fatnet"])
            # A resumed refinement run may intentionally request a new LR.
            # Optimizer state loading restores the old LR, so override it with
            # the current command-line value after loading compatible state.
            for param_group in optimizer_sam.param_groups:
                param_group["lr"] = args.lr
            for param_group in optimizer_fatnet.param_groups:
                param_group["lr"] = args.lr
            logging.info(
                f"Resumed from {args.resume} at epoch {start_epoch}; "
                f"optimizer learning rate reset to {args.lr}"
            )
    scheduler_sam = None
    scheduler_fatnet = None
    if args.lr_schedule == "cosine":
        schedule_epochs = max(1, num_epochs - start_epoch)
        scheduler_sam = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_sam, T_max=schedule_epochs, eta_min=args.min_lr
        )
        scheduler_fatnet = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_fatnet, T_max=schedule_epochs, eta_min=args.min_lr
        )
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()  # 记录开始时间
        epoch_loss = 0
        for step, (image, gt2D, boxes, _) in enumerate(tqdm(train_dataloader)): 
            optimizer_sam.zero_grad()
            optimizer_fatnet.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if args.use_amp:
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = medsam_model(image, boxes_np)
                    loss = 0.5 * seg_loss(medsam_pred, gt2D) + 0.5 * ce_loss(medsam_pred, gt2D.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer_sam)
                scaler.step(optimizer_fatnet)
                scaler.update()
                optimizer_sam.zero_grad()
                optimizer_fatnet.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np)
                loss = 0.5 * seg_loss(medsam_pred, gt2D) + 0.5 * ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer_sam.step()
                optimizer_fatnet.step()
                optimizer_sam.zero_grad()
                optimizer_fatnet.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step
        losses.append(epoch_loss)

        end_time = time.time()  # 记录结束时间
        epoch_duration = end_time - start_time  # 计算一个epoch的耗时

        logging.info(
            f'Epoch Time: {epoch_duration:.2f} seconds'
        )

        # 验证模型并计算验证损失
        val_loss = validate(medsam_model, val_dataloader, seg_loss, ce_loss)
        val_losses.append(val_loss)  # 添加验证损失

        # scheduler.step(val_loss)

        logging.info(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, '
            f'Train Loss: {epoch_loss}, Val Loss: {val_loss}, '
            f'LR: {optimizer_fatnet.param_groups[0]["lr"]}'
        )


        ## save the latest model
        checkpoint = {
            "model": medsam_model.state_dict(),
            "optimizer_sam": optimizer_sam.state_dict(),
            "optimizer_fatnet": optimizer_fatnet.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer_sam": optimizer_sam.state_dict(),
                "optimizer_fatnet": optimizer_fatnet.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        if scheduler_sam is not None:
            scheduler_sam.step()
            scheduler_fatnet.step()

        # %% plot train and validation loss
        plt.plot(losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, args.task_name + "_train_val_loss.png"))
        plt.close()


if __name__ == "__main__":
    main()
