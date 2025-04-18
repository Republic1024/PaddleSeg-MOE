import os
import numpy as np
from PIL import Image
import subprocess
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import seaborn as sns
from pylab import mpl
from skimage import io

# 配置 Notebook 的字体和风格
warnings.filterwarnings("ignore")
import numpy as np

# 构造类别编号到 RGB 的映射（索引即为类别编号）
id2color = np.array(
    [
        [0, 0, 0],  # 0 - 背景
        [220, 20, 60],  # 1 - 建筑
        [128, 64, 128],  # 2 - 道路
        [0, 0, 255],  # 3 - 水体
        [210, 180, 140],  # 4 - 裸土
        [34, 139, 34],  # 5 - 林地
        [255, 255, 0],  # 6 - 耕地
    ],
    dtype=np.uint8,
)

from PIL import Image
import numpy as np


def overlay_mask_on_image(
        img: np.ndarray, mask: np.ndarray, colormap: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """
    使用 PIL 实现透明叠加：将 mask 映射为彩色图，并叠加到原图上。

    参数:
        img (np.ndarray): 原始图像（RGB），shape = (H, W, 3)
        mask (np.ndarray): 分割掩码（0-6的类别编号），shape = (H, W)
        colormap (np.ndarray): 类别编号到 RGB 的映射表，shape = (N, 3)
        alpha (float): 叠加透明度（0-1）

    返回:
        np.ndarray: 叠加后的图像（uint8, RGB）
    """
    # 映射 mask -> RGB 彩色图
    mask_rgb = colormap[mask]  # shape = (H, W, 3)

    # 转为 float 做加权
    img_f = img.astype(np.float32)
    mask_f = mask_rgb.astype(np.float32)

    # 透明叠加
    blended = (1 - alpha) * img_f + alpha * mask_f
    blended = blended.clip(0, 255).astype(np.uint8)

    return blended


# 🚀 多分类分割函数（全图多标签推理）
def all_seg(
        image_path: str,
        config_path: str = "configs/segmenter/segmenter_rural2.yml",
        model_path: str = "../output/segmenter_rural2/best_model/model.pdparams",
        save_dir: str = "../output/test",
) -> np.ndarray:
    """
    使用 PP-LiteSeg 多分类模型对输入图像进行分割预测。

    参数:
        image_path (str): 输入图像路径
        config_path (str): PaddleSeg 配置文件路径
        model_path (str): PaddleSeg 模型权重路径
        save_dir (str): 结果输出路径

    返回:
        mask_np (np.ndarray): 分割输出的 mask 数组
    """
    cmd = [
        "python",
        "tools/predict.py",
        "--config",
        config_path,
        "--model_path",
        model_path,
        "--image_path",
        image_path,
        "--save_dir",
        save_dir,
    ]

    print(f"[🚀] 正在用 PP-LiteSeg 推理图像（多分类）：{image_path}")
    subprocess.run(cmd, check=True)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(save_dir, "pseudo_color_prediction", f"{image_name}.png")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"[❌] 未找到预测输出文件：{mask_path}")
    else:
        print(f"[✅] 成功读取输出文件：{mask_path}")

    mask_img = Image.open(mask_path)
    return np.array(mask_img)


# 🚗 道路单类别分割函数（仅提取 road 类）
def road_seg(
        image_path: str,
        config_path: str = "configs/road_seg/pp_liteseg_stdc1_deepglobe_infer.yml",
        model_path: str = "pp_liteseg_stdc1_deepglobe.pdparams",
        save_dir: str = "../output/test",
) -> np.ndarray:
    """
    使用 PP-LiteSeg 道路分割模型对图像进行推理。

    参数:
        image_path (str): 输入图像路径
        config_path (str): 模型配置文件路径
        model_path (str): 模型权重路径
        save_dir (str): 输出目录

    返回:
        mask_np (np.ndarray): 推理后道路掩码图像数组
    """
    print(f"[🚗] 正在用 PP-LiteSeg 道路模型推理图像：{image_path}")
    cmd = [
        "python",
        "tools/predict.py",
        "--config",
        config_path,
        "--model_path",
        model_path,
        "--image_path",
        image_path,
        "--save_dir",
        save_dir,
    ]

    subprocess.run(cmd, check=True)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(save_dir, "pseudo_color_prediction", f"{image_name}.png")

    if not os.path.exists(output_path):
        print(f"[❌] 道路分割输出未生成：{output_path}")
        return None
    else:
        print(f"[✅] 成功读取道路分割结果：{output_path}")
        mask_img = Image.open(output_path)
        return np.array(mask_img)


import os
from skimage import io, transform, img_as_ubyte


# def resize_and_save(img_path, size=(512, 512)):
#     """
#     读取图像并 resize 到固定大小，然后以新的文件名保存。
#
#     参数：
#         img_path (str): 原始图像路径
#         size (tuple): 新的图像大小，默认 (512, 512)
#
#     返回：
#         new_path (str): resize 后图像的保存路径
#     """
#     # 读取图像
#     img = io.imread(img_path)
#
#     # Resize 到指定大小（保持图像通道）
#     img_resized = transform.resize(img, size, preserve_range=True, anti_aliasing=True)
#     img_resized = img_as_ubyte(img_resized)  # 转换为uint8格式
#
#     # 构造新路径
#     base, ext = os.path.splitext(img_path)
#     new_path = f"{base}_512_512.png"
#
#     # 保存图像
#     io.imsave(new_path, img_resized)
#
#     return new_path


def resize_and_save(img_path, size=(512, 512)):
    """
    读取图像并 resize 到固定大小，然后以新的文件名保存。

    参数：
        img_path (str): 原始图像路径
        size (tuple): 新的图像大小，默认 (512, 512)

    返回：
        new_path (str): resize 后图像的保存路径
    """
    # 读取图像
    img = io.imread(img_path)

    # Resize 到指定大小（保持通道，且结果为 float64 in [0, 1]）
    img_resized = transform.resize(img, size, preserve_range=True, anti_aliasing=True)

    # 转换为 uint8 格式 (0~255)
    img_resized_uint8 = img_as_ubyte(img_resized)

    # 构造新路径
    base, ext = os.path.splitext(img_path)
    new_path = f"{base}_512_512.png"

    # 保存图像
    io.imsave(new_path, img_resized_uint8)

    return new_path


import os
import numpy as np
from skimage import io, transform, img_as_ubyte


def resize_and_save_2(img_path, size=(512, 512)):
    """
    读取图像并 resize 到指定大小，再保存为带 _512_512 后缀的图片。

    参数：
        img_path (str): 原始图像路径
        size (tuple): 目标尺寸，如 (512, 512)

    返回：
        new_path (str): 保存后的图像路径
    """
    # 1. 读取图像
    img = io.imread(img_path)

    # 2. Resize 到目标尺寸（保持通道）
    img_resized = transform.resize(img, size, preserve_range=False, anti_aliasing=True)  # 返回 float in [0, 1]

    # 3. 转换为 uint8 格式（img_as_ubyte 要求值必须在 [0, 1]）
    img_uint8 = img_as_ubyte(img_resized)

    # 4. 构造新路径
    base, ext = os.path.splitext(img_path)
    new_path = f"{base}_{size[0]}_{size[1]}.png"

    # 5. 保存图像
    io.imsave(new_path, img_uint8)

    return new_path

# # 示例调用（推荐放在 notebook 的代码单元中）
# test_image = "../photo/1160.png"
# road_mask = road_seg(test_image)
# mask = all_seg(test_image)
# mask_2 = mask.copy()

# 🚀 示例调用（推荐放在 notebook 的代码单元中）

# test_image = "../photo/1173.png"

# # ✅ 道路分割模型（只分出 road 类）
# road_mask = road_seg(
#     image_path=test_image,
#     config_path="configs/road_seg/pp_liteseg_stdc1_deepglobe_infer.yml",
#     model_path="pp_liteseg_stdc1_deepglobe.pdparams",
#     save_dir="../output/test",
# )

# # ✅ 多类地物分割模型（输出0~6的语义mask）
# mask = all_seg(
#     image_path=test_image,
#     config_path="configs/segmenter/segmenter_rural2.yml",
#     model_path="../output/segmenter_rural2/best_model/model.pdparams",
#     save_dir="../output/test",
# )

# # ✅ 可选备份一份副本（用于后续处理）
# mask_2 = mask.copy()
# mask[road_mask == 1] = 2  # 2: road

# # 读取图像（确保是 RGB）
# img = np.array(Image.open(test_image).convert("RGB"))

# # 假设 mask 已是 0~6 的类别编号矩阵
# # mask = ...

# # 类别映射（前面定义的 id2color）
# colored_overlay = overlay_mask_on_image(img, mask, id2color, alpha=0.3)

# io.imshow(colored_overlay)
