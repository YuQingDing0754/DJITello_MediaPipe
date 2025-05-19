# Tello无人机人体跟踪系统

一个复杂的计算机视觉系统，使DJI Tello无人机能够自主检测、跟踪和跟随人体，并可通过手势识别触发特殊动作。

## 主要功能

- **实时人体检测与跟踪：** 使用MediaPipe姿态检测技术识别和跟踪目标人体
- **智能跟随：** 通过自适应PID控制实现平滑跟随目标人物
- **手势识别控制：**
  - **右手挥手：** 命令无人机降落
  - **左手挥手：** 触发特殊空中机动（眼镜蛇机动）
  - **剪刀手（✌️）：** 拍照并执行后空翻
- **位置预测：** 预测目标移动轨迹，即使在短暂丢失目标时仍能保持跟踪
- **障碍物避免：** 结合视觉检测和ToF传感器避开障碍物
- **性能优化：** 自适应帧处理和多线程架构确保运行流畅
- **中文文本显示：** 带有正确中文渲染的屏幕状态信息

## 系统需求

### 硬件
- DJI Tello无人机
- 具备Wi-Fi功能的电脑（需连接到Tello的网络）
- 至少720p分辨率的摄像头（使用Tello内置摄像头）

### 软件依赖
- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- MediaPipe
- djitellopy
- PIL (Python图像处理库)
- Threading和concurrent.futures (Python标准库)

## 安装步骤

1. 克隆本仓库：
   ```bash
   git clone https://github.com/yourusername/tello-human-tracking.git
   cd tello-human-tracking
   ```

2. 安装所需依赖：
   ```bash
   pip install opencv-python numpy mediapipe djitellopy pillow
   ```

3. 确保安装了适当的中文字体，以便显示中文文本。系统将在以下位置查找字体：
   - 当前目录：`simhei.ttf`
   - Windows系统：`C:/Windows/Fonts/simhei.ttf`、`C:/Windows/Fonts/simkai.ttf`等
   - Linux系统：`/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`
   - Mac系统：`/System/Library/Fonts/STHeiti Light.ttc`

## 使用方法

1. 将电脑连接到Tello无人机的Wi-Fi网络

2. 运行主程序：
   ```bash
   python tello_human_tracking.py
   ```

3. 无人机将自动起飞并开始扫描寻找人体目标

4. 一旦检测到人，无人机将跟踪并跟随该人

5. 使用以下手势控制：
   - 左右来回挥动右手3次，命令无人机降落
   - 左右来回挥动左手3次，触发特殊空中机动
   - 做出剪刀手（✌️）手势，拍照并执行后空翻

6. 按键盘上的`q`键退出程序并让无人机降落

### 简易模式

如果您喜欢更简单的版本，仅聚焦于头部检测和剪刀手拍照功能：

```bash
python tello_simple_photo.py
```

简易版本功能：
1. 检测人的头部
2. 当检测到头部时锁定位置
3. 当检测到剪刀手时拍照
4. 执行后空翻以表示照片已拍摄

## 键盘控制

- `q`：退出程序并让无人机降落
- `r`：重置目标跟踪
- `p`：切换位置预测功能
- `o`：切换障碍物避免功能
- `t`：手动触发特殊机动
- `ESC`：紧急降落

## 配置参数

您可以在脚本顶部调整各种参数：

### 性能设置
```python
DISPLAY_FPS = 60                # 目标显示帧率
PROCESS_EVERY_N_FRAMES = 3      # 每N帧处理一次（降低处理负载）
MAX_PROCESSING_TIME = 0.1       # 最大处理时间阈值（秒）
LOW_RES_PROCESSING = True       # 使用低分辨率进行处理
PROCESS_RESOLUTION = (480, 360) # 处理分辨率
DISPLAY_RESOLUTION = (960, 720) # 显示分辨率
```

### PID控制参数
```python
pid_x = [0.1, 0.1, 0]  # 左/右移动
pid_y = [0.4, 0.4, 0]  # 上/下移动
pid_z = [0.4, 0.4, 0]  # 前/后移动
pid_yaw = [0.6, 0.4, 0]  # 旋转
```

### 安全设置
```python
obstacle_detection_enabled = True
obstacle_safety_distance = 50  # 障碍物安全距离（像素）
person_safety_distance = 100  # 其他人安全距离（像素）
min_safe_tof_distance = 60  # 最小安全ToF距离（厘米）
```

## 常见问题排查

### 常见问题

1. **无人机无法连接：**
   - 确保您已连接到Tello的Wi-Fi网络
   - 检查无人机电量是否充足（>15%）

2. **跟踪性能不佳：**
   - 改善光线条件
   - 降低`PROCESS_RESOLUTION`以加快处理
   - 增加`PROCESS_EVERY_N_FRAMES`以跳过更多帧

3. **手势识别问题：**
   - 在良好光线下清晰做出手势
   - 确保您的手和手臂完全可见
   - 尝试调整代码中的挥手阈值参数

4. **中文文本显示不正确：**
   - 验证是否安装了兼容的中文字体
   - 检查`cv2_add_chinese_text`函数中的字体路径

5. **电池消耗过快：**
   - 通过调整PID值减少不必要的移动
   - 如不需要则禁用障碍物检测
---

**注意**：在您所在地区，无人机飞行可能受到法规限制。操作无人机时请始终遵守当地法律法规。
