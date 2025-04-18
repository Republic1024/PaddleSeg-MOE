import os
import subprocess
from skimage import io

# 全局设定
SAVE_DIR = r"output/test"
PSEUDO_DIR = os.path.join(SAVE_DIR, "pseudo_color_prediction")


def infer_single(image_path):
    """
    使用 PP-LiteSeg 模型对单张图像进行推理，返回预测后的 mask np.ndarray。
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"[🚀] 正在用 PP-LiteSeg 推理图像：{image_name}")

    cmd = [
        "python", "tools/predict.py",
        "--config", "configs/road_seg/pp_liteseg_stdc1_deepglobe_infer.yml",
        "--model_path", "pp_liteseg_stdc1_deepglobe.pdparams",
        "--image_path", image_path,
        "--save_dir", SAVE_DIR
    ]
    subprocess.run(cmd)
    print(cmd)
    output_path = os.path.join(PSEUDO_DIR, f"{image_name}.png")
    print(output_path)
    if os.path.exists(output_path):
        print(f"[✅] 输出文件读取：{output_path}")
        mask_np = io.imread(output_path)
        return mask_np
    else:
        print(f"[❌] 未找到输出文件：{output_path}")
        return None


def infer_seg(image_path):
    """
    使用 SegFormer 模型对单张图像进行推理，返回预测后的 mask np.ndarray。
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"[🚀] 正在用 SegFormer 推理图像：{image_name}")

    cmd = [
        "python", "tools/predict.py",
        "--config", "configs/segformer/segformer_udd_b3.yml",
        "--model_path", "output/segformer_udd_40000/model.pdparams",
        "--image_path", image_path,
        "--save_dir", SAVE_DIR
    ]
    subprocess.run(cmd)

    output_path = os.path.join(PSEUDO_DIR, f"{image_name}.png")
    if os.path.exists(output_path):
        print(f"[✅] 输出文件读取：{output_path}")
        mask_np = io.imread(output_path)
        return mask_np
    else:
        print(f"[❌] 未找到输出文件：{output_path}")
        return None


def infer_pplite(image_path):
    """
    使用 PP-LiteSeg WHDLD 模型推理单张图像，返回预测结果的 numpy mask。
    """
    save_dir = "./output/pplite_seg_whdld_3"
    pseudo_dir = os.path.join(save_dir, "pseudo_color_prediction")
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 构造命令
    cmd = [
        "python", "tools/predict.py",
        "--config", "configs/pp_liteseg/pp_liteseg_whdld_3.yml",
        "--model_path", "output/pplite_seg_whdld_3/best_model_2/model.pdparams",
        "--image_path", image_path,
        "--save_dir", save_dir,
        "--crop_size", "256", "256",
        "--stride", "128", "128"
    ]

    # 执行命令
    print(f"[🚀] 正在推理：{image_name}")
    subprocess.run(cmd)

    # 结果路径
    mask_path = os.path.join(pseudo_dir, f"{image_name}.png")
    if os.path.exists(mask_path):
        mask_np = io.imread(mask_path)
        print(f"[✅] Mask 成功读取：{mask_path}")
        return mask_np
    else:
        print(f"[❌] 未找到输出文件：{mask_path}")
        return None


# # 示例调用（可注释）
# if __name__ == "__main__":
#     test_image = r"C:\Users\Administrator\Pictures\2.jpg"
#     mask_single = infer_single(test_image)
#     print(mask_single)
#     mask_pplite = infer_pplite(test_image)
#     print(mask_pplite)
