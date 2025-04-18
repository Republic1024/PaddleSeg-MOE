## 🌏 遥感图像分割项目（MOE结构 | PaddleSeg实现）

本项目基于飞桨官方图像分割套件 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)，实现了遥感图像分割任务，并引入 **MOE（Mixture of Experts）结构** 以增强模型的泛化能力与任务适应性。


---

以下是你这部分内容的 **润色优化版**，我对格式、逻辑顺序和表达方式做了完善，使其更清晰专业，并适配 `README.md` 的排版规范：

---

## 🛠 部署环境

本项目基于以下环境开发与运行：

- Python：`3.9.20`
- PaddlePaddle-GPU：`2.5.1`
- 推荐使用 Anaconda 或 Miniconda 管理环境

---

## 🚀 快速开始

### 1️⃣ 克隆项目

```bash
git clone https://github.com/Republic1024/PaddleSeg3.git
cd PaddleSeg3
```

### 2️⃣ 下载模型参数（checkpoint）

请从以下链接下载训练好的模型参数（`output/` 文件夹）并替换项目中的空文件夹：

```
https://pan.baidu.com/s/14FohHLISAdQJCgr2NoKaoQ?pwd=rryh 
提取码: rryh 
```

下载后，将 `output/` 文件夹覆盖到本地 `PaddleSeg3/` 项目目录下。

---

### ✅ 创建 Python 环境（推荐使用 Conda）

```bash
# 创建名为 ps 的 Conda 环境
conda create -n ps python=3.9

# 激活环境
conda activate ps

# 安装 PaddlePaddle（根据你的 CUDA 版本）
# CUDA 12.0
python -m pip install paddlepaddle-gpu==2.5.2.post120 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
```

📌 请根据你的 CUDA 驱动版本选择正确的安装源，详见：https://www.paddlepaddle.org.cn/install/old
---

### 🧩 安装项目依赖

确保项目根目录存在 `requirements.txt`，执行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

---

## 🧪 运行最小 DEMO

进入核心代码目录并运行示例：

```bash
cd paddleseg
```
打开并运行 `road_seg.ipynb` Notebook 文件，进行道路分割演示。

---


## 🔧 项目结构说明

项目主要包含以下模块：

- `configs/`：配置文件目录，包含模型结构、优化器和数据处理配置
- `paddleseg/`：分割算法主逻辑与网络结构
- `tools/`：训练与推理脚本
- `output/`：模型保存路径与推理结果（此版本已去除大文件，仅保留结构）
- `*.ipynb` / `*.py`：示例脚本与调试流程

---

## 🧠 多专家结构任务流程（MOE）

### ✅ 多类地物分割模型（输出0~6的语义 mask）

以下代码基于自定义 `all_seg()` 接口完成一次全图预测，输出为 0~6 语义标签：

```python
mask = all_seg(
    image_path=img_path,
    config_path="configs/segmenter/segmenter_rural2.yml",
    model_path="../output/segmenter_rural2/best_model/model.pdparams",
    save_dir="../output/test",
)
```

---

### 🌲 森林分割专家（基于 Segmenter）

为进一步提升特定类别（如“森林”）的分割效果，集成森林专家模型：

```python
forest_mask = all_seg(
    image_path=img_path,  # --image_path 参数
    config_path="./configs/segformer/segformer_udd_b3.yml",  # --config 参数
    model_path=r"..\output\iter_40000\model.pdparams",  # --model_path 参数
    save_dir="../output/test",  # --save_dir 参数
)
```

---

### 🧩 专家嵌入逻辑（融合道路与森林专家输出）

MOE结构的关键在于融合不同专家的结果，示例如下：

```python
mask[road_mask == 1] = 2      # 类别2：道路
mask[forest_mask == 0] = 5    # 类别5：森林
```

---

## 🧪 预测案例

以下为基于 `RtFormer` 网络结构的遥感图像预测示例：

```bash
cd paddleseg
python tools/predict.py \
  --config configs/rtformer/rtformer_base_cityscapes_1024x512_120k.yml \
  --model_path ../output/rtformer_udd/best_model_2/model.pdparams \
  --image_path "../photo/DJI_00527.JPG" \
  --save_dir ../output/rtformer_udd
```

以下基于PPLITESEG rural为例，对模型进行训练：
```bash
python tools/train.py --config configs/pp_liteseg/pp_liteseg_rural_2.yml --save_dir output/rural_seg_pplite_2 --save_interval 500 --do_eval 
```
---

## 📌 注意事项

- 请确保 `.pdparams` 权重路径与配置一致
- 当前版本为**代码精简结构骨架**，未包含大文件与完整权重
- 若需完整模型与训练日志，请单独获取或重新训练



