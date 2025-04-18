## 🌏 遥感图像分割项目（MOE结构 | PaddleSeg实现）

本项目基于飞桨官方图像分割套件 [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)，实现了遥感图像分割任务，并引入 **MOE（Mixture of Experts）结构** 以增强模型的泛化能力与任务适应性。

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
    model_path=r"D:\pythonProject\DeepSeek\ps3\PaddleSeg\output\iter_40000\model.pdparams",  # --model_path 参数
    save_dir="../output/test",  # --save_dir 参数
)
```

---

### 🧩 专家嵌入逻辑（融合道路与森林专家输出）

MOE结构的关键在于融合不同专家的结果，示例如下：

```python
mask_2 = mask.copy()
mask[road_mask == 1] = 2      # 类别2：道路
mask[forest_mask == 0] = 5    # 类别5：森林
```

---

## 🧪 预测案例

以下为基于 `RtFormer` 网络结构的遥感图像预测示例：

```bash
python tools/predict.py \
  --config configs/rtformer/rtformer_base_cityscapes_1024x512_120k.yml \
  --model_path ../output/rtformer_udd/best_model_2/model.pdparams \
  --image_path "../photo/DJI_00527.JPG" \
  --save_dir ../output/rtformer_udd
```

---

## 📌 注意事项

- 请确保 `.pdparams` 权重路径与配置一致
- 当前版本为**代码精简结构骨架**，未包含大文件与完整权重
- 若需完整模型与训练日志，请单独获取或重新训练



