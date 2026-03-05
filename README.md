## 项目说明：Workout Form Analyzer（期末项目）

本项目是一个**实时健身动作姿势分析系统**，使用 MediaPipe 捕捉人体关键点，并通过自编码器（Autoencoder）学习“正确动作”的关节角度分布。当你的动作姿势偏离理想状态时，系统会显示一个**纠正用的“幽灵骨架”**，提示你应该如何调整。

目前仓库中已经包含一个参考示例：`test_models/claude_guide/`，里面实现了一个完整可运行的 Demo（下文会说明如何运行）。

> ⚠️ 说明：下面的说明是一个通用模板，你可以在熟悉项目后按需要补充/修改具体内容（例如你自己的模型、报告链接等）。

---

## 环境要求

- 操作系统：推荐在 **WSL2 下的 Ubuntu** 或原生 Linux / macOS 上运行
- Python：建议 **Python 3.10.x**
- 显卡：如果需要加速训练，建议有支持 CUDA 的 NVIDIA GPU（非必须）
- 摄像头：用于实时动作捕捉（USB 摄像头或笔记本自带摄像头）

---

## 目录结构（核心示例）

项目根目录下与模型示例相关的部分大致如下（参考）：

```bash
test_models/
  └── claude_guide/
      ├── form_analyzer.py        # 实时动作分析主程序
      ├── extract_landmarks.py    # 从视频中提取关节角数据
      ├── train_autoencoder.py    # 训练自编码器模型
      ├── angles.csv              # 训练数据（关节角）
      ├── models/                 # 训练好的模型文件
      └── README.md               # 该子项目的英文说明
```

> 提示：如果你在根目录新增了自己的代码（例如 Web 前端、额外脚本等），建议在这里补充项目结构说明，方便之后查阅。

---

## 快速开始：运行示例程序

以下步骤以 `test_models/claude_guide/` 目录为例，演示如何在 WSL Ubuntu 中运行**实时姿势分析 Demo**。

### 1. 进入项目目录

在 WSL 终端中执行：

```bash
cd ~/final_project/project_shared
cd test_models/claude_guide
```

### 2. 创建并激活虚拟环境（推荐）

使用 `venv`（系统自带，无需额外安装）：

```bash
python3.10 -m venv ai_form_v2
source ai_form_v2/bin/activate
```

或者如果你使用 `pyenv`，可以参考子目录 README 中的写法：

```bash
pyenv virtualenv 3.10.6 ai_form_v2
pyenv activate ai_form_v2
```

### 3. 安装依赖

在激活虚拟环境后，安装依赖（确保当前目录为 `test_models/claude_guide/`）：

```bash
pip install -r requirements.txt
```

### 4. 运行实时姿势分析程序

```bash
python form_analyzer.py
```

确保：

- 你的摄像头已连接并可被系统识别；
- `models/` 目录与 `form_analyzer.py` 在同一目录下（默认就是如此）。

#### 交互按键

运行时窗口获得焦点后，可以使用：

- `S`：切换到 **深蹲（squat）模式**
- `L`：切换到 **侧平举（lateral raise）模式**
- `Q`：退出程序

---

## 训练你自己的模型（可选）

如果你希望使用自己的动作数据重新训练模型，可以参考下面的流程（与 `test_models/claude_guide/README.md` 一致）：

1. **收集视频数据**  
   准备多段“动作标准”的视频，例如：
   ```bash
   videos/
   ├── squat/
   │   ├── video1.mp4
   │   └── video2.mp4
   └── lateral raise/
       ├── video1.mp4
       └── video2.mp4
   ```

2. **提取关节角数据**
   ```bash
   python extract_landmarks.py
   ```
   运行结束后，会在当前目录生成/更新 `angles.csv`。

3. **训练自编码器**
   ```bash
   python train_autoencoder.py
   ```
   训练完成后，新的模型会保存在 `models/` 目录中，供 `form_analyzer.py` 使用。

---

## 依赖与版本说明

详细依赖请参考 `test_models/claude_guide/requirements.txt`，其中几个关键包为：

- `mediapipe`：人体姿势关键点检测
- `tensorflow`：自编码器模型训练与推理
- `opencv-python`：摄像头采集与画面渲染
- `scikit-learn`：特征归一化 / 标准化
- `numpy`、`pandas`：数据处理

> 注意：MediaPipe 与 TensorFlow 某些版本的 `protobuf` 依赖存在冲突问题，建议使用 `requirements.txt` 中已经验证过的版本组合，并使用 Python 3.10。

---

## 常见问题（FAQ）

**Q1：运行 `form_analyzer.py` 时摄像头打不开怎么办？**  
- 确认系统中摄像头正常工作（可在其他应用中测试）；  
- 在 WSL 环境下，需确保你已正确配置了摄像头透传；  
- 可以尝试修改代码中 `cv2.VideoCapture()` 的设备索引（例如从 `0` 改成 `1`、`2`）。

**Q2：画面很卡 / 延迟很高？**  
- 关闭其他占用大量显卡/CPU 的程序；  
- 将窗口缩小、或降低分辨率；  
- 如有 NVIDIA GPU，可考虑安装 GPU 版 TensorFlow 并配置 CUDA 环境。

**Q3：如何在根目录运行整个项目？**  
- 目前核心可运行代码在 `test_models/claude_guide/` 下；  
- 如果你之后在根目录增加统一入口（例如 `main.py`、Web 前端等），建议在此 README 的“快速开始”部分补充具体命令。

---

## 后续可补充内容（建议）

你完成期末项目后，可以在本 README 中进一步补充：

- 项目背景与目标说明（为什么要做这个系统）  
- 模型设计与训练细节（网络结构、损失函数、评价指标等）  
- 实验结果（图表、截图、效果对比）  
- 未来改进方向（支持更多动作、多摄像头、3D 姿势等）

这样不仅方便自己复盘，也方便老师/同学快速理解和运行你的项目。
