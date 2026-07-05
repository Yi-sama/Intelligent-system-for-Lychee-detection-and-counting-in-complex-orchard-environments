# 复杂果园环境下荔枝检测与计数智能系统

这是一个基于 PyQt5 的桌面端项目，用于复杂果园环境下的荔枝目标检测、计数与结果导出。系统将图片检测、批量图片处理、本地视频检测、摄像头预览与检测、训练脚本支持整合在同一个 Windows 本地工作流中。

## 项目状态

当前仓库对应的是一个已经完成主要功能、可用于课程展示与演示的桌面原型系统。

已实现内容包括：

- 单张图片检测
- 批量图片检测
- 本地视频逐帧检测
- 本地摄像头预览与实时检测基础流程
- CSV 结果导出
- 标注结果图片导出
- 标注结果视频导出
- YOLO 训练启动脚本
- 针对推理、导出、GUI 布局、批处理、视频流程、摄像头流程的测试覆盖

## 主要功能

- 提供图片、视频、摄像头三种模式的统一 PyQt5 桌面界面
- 支持置信度与 IoU 阈值调节
- 采用以预览区域为核心的交互布局
- 显示检测结果汇总与逐目标细节信息
- 支持文件夹批量图片检测
- 支持视频检测进度展示与结果回放
- 支持摄像头预览及设备释放处理
- 支持结构化 CSV 结果导出

## 项目结构

```text
app/                    桌面 GUI、推理服务、导出逻辑
tests/                  自动化测试
docs/                   进度记录、调试记录、项目说明
Reference/              基础模型、依赖文件、参考资源
outputs/                推理输出目录
lychee dataset/         本地训练/推理数据集目录（已被 Git 忽略）
start.bat               主界面启动脚本
start_new.bat           新界面启动脚本
train.bat               训练启动脚本
```

## 运行环境

- 操作系统：Windows
- 推荐 Python 环境：`D:\Miniconda3\envs\yolo`
- 核心依赖：
  - `PyQt5`
  - `ultralytics`
  - `opencv-python`
  - `numpy`
  - `Pillow`
  - `torch`

依赖安装参考：

```powershell
pip install -r Reference\requirements.txt
```

## 快速开始

1. 准备好 Conda 环境与依赖。
2. 确认默认模型文件存在：`Reference\yolo26n.pt`
3. 启动桌面程序：

```powershell
start.bat
```

也可以直接执行：

```powershell
conda run -p D:\Miniconda3\envs\yolo python -m app.main
```

如果需要启动新的界面版本，可以使用：

```powershell
start_new.bat
```

## 训练说明

训练流程独立于 GUI 执行：

```powershell
train.bat
```

当前训练配置：

- 基础模型：`Reference\yolo26n.pt`
- 数据配置：`lychee dataset\combined-detect.yaml`
- 输出目录：`runs\lychee\yolo26_merged_detect_safe`

期望生成的训练权重：

```text
runs\lychee\yolo26_merged_detect_safe\weights\best.pt
```

完成训练后，可以将得到的权重替换默认通用模型，用于更加贴合本项目的荔枝检测推理。

## 测试与校验

推荐在目标环境中运行测试：

```powershell
conda run -p D:\Miniconda3\envs\yolo python -m unittest tests.test_app
```

轻量语法校验可使用：

```powershell
python -m py_compile app\gui.py app\infer.py app\export.py tests\test_app.py
```

## 说明

- 仓库默认忽略本地数据集、训练输出、生成结果以及过程性工具文件。
- 当前工作流面向本地 Windows + Conda 环境优化。
- 启动脚本中的默认环境路径是本机路径，迁移到其他设备时需要按实际环境调整。

