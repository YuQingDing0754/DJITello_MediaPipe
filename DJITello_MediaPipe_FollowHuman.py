# COPYRIGHT YUQING DING (Scott)
import cv2
import numpy as np
import time
from djitellopy import Tello
import mediapipe as mp
from collections import deque
import math
from PIL import Image, ImageDraw, ImageFont
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

# 性能优化参数
DISPLAY_FPS = 60                # 目标显示帧率
PROCESS_EVERY_N_FRAMES = 3      # 每N帧处理一次 (降低处理负载)
MAX_PROCESSING_TIME = 0.1       # 最大处理时间阈值 (秒)
LOW_RES_PROCESSING = True       # 使用低分辨率进行处理
PROCESS_RESOLUTION = (480, 360) # 处理分辨率
DISPLAY_RESOLUTION = (960, 720) # 显示分辨率

# 帧处理和缓冲
frame_buffer = deque(maxlen=5)    # 原始帧缓冲
processed_frames = queue.Queue(maxsize=10)  # 处理后的帧队列
latest_landmarks = None            # 最新的姿态关键点
processing_active = True           # 处理线程活动标志
display_active = True              # 显示线程活动标志
fps_stats = deque(maxlen=30)       # FPS统计

# 跟随参数
pid_x = [0.1, 0.1, 0]
pid_y = [0.4, 0.4, 0]
pid_z = [0.4, 0.4, 0]
pid_yaw = [0.6, 0.4, 0]
pError_x, pError_y, pError_z, pError_yaw = 0, 0, 0, 0

# 人体位置历史 - 用于预测
position_history = deque(maxlen=15)  # 增加历史记录长度
velocity_history = deque(maxlen=10)  # 用于存储速度
last_movement_direction = [0, 0]

# 目标人体跟踪
target_id = None
target_position = None
target_lost_frames = 0
MAX_LOST_FRAMES = 30
prediction_enabled = True  # 启用预测功能

# 右手挥手检测参数
right_wrist_history = deque(maxlen=20)  # 存储右手腕位置历史
wave_threshold = 25  # 挥手水平移动阈值 (像素)
right_wave_count = 0  # 检测到的右手挥手次数
right_wave_direction = 0  # 挥手方向 (-1:左, 1:右, 0:无)
REQUIRED_WAVES = 3  # 需要连续检测到的挥手次数
right_wave_detected = False  # 是否检测到完整的右手挥手动作
right_wave_cooldown = 0  # 挥手检测冷却时间(避免误触)

# 左手挥手检测参数
left_wrist_history = deque(maxlen=20)  # 存储左手腕位置历史
left_wave_count = 0  # 检测到的左手挥手次数
left_wave_direction = 0  # 挥手方向 (-1:左, 1:右, 0:无)
left_wave_detected = False  # 是否检测到完整的左手挥手动作
left_wave_cooldown = 0  # 挥手检测冷却时间(避免误触)
performing_stunt = False  # 是否正在执行特技动作
stunt_position = None  # 特技动作前的位置
# 手势检测相关参数
mp_hands = mp.solutions.hands
hands_results = None
hands_processing_active = True
peace_sign_detected = False
peace_sign_cooldown = 0
PEACE_SIGN_COOLDOWN = 30
performing_photo_stunt = False

# 避障参数
obstacle_detection_enabled = True
obstacle_safety_distance = 50  # 障碍物安全距离 (像素)
person_safety_distance = 100  # 其他人安全距离 (像素)
obstacle_regions = [False, False, False, False, False]  # 左上中右下区域是否有障碍物
tof_distance = 0  # ToF传感器距离 (厘米)
min_safe_tof_distance = 60  # 最小安全ToF距离 (厘米)

# 防止自动降落参数
MIN_CONTROL_VALUE = 5
last_command_time = time.time()
MAX_COMMAND_INTERVAL = 0.3
keepalive_counter = 0

# 移动模式标志
ROTATION_PREFERENCE = 0.8

# 支持中文显示 - 增强版本
def cv2_add_chinese_text(img, text, position, textColor=(0, 255, 0), textSize=30):
    """添加中文文本到图像上，确保使用SimHei字体正确渲染"""
    if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # 创建一个可以在给定图像上绘制的对象
    draw = ImageDraw.Draw(img)
    
    # 尝试多种方式找到合适的字体
    font_path = None
    # 指定的字体路径列表
    font_paths = [
        "simhei.ttf",                      # 当前目录
        "C:/Windows/Fonts/simhei.ttf",     # Windows 标准位置
        "C:/Windows/Fonts/simkai.ttf",     # 楷体备选
        "C:/Windows/Fonts/simsun.ttc",     # 宋体备选
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # Linux 可能位置
        "/System/Library/Fonts/STHeiti Light.ttc"  # Mac 可能位置
    ]
    
    # 尝试找到第一个可用的字体
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break
    
    # 如果找不到字体文件，则从字节生成一个基本字体
    if font_path:
        fontStyle = ImageFont.truetype(font_path, textSize, encoding="utf-8")
    else:
        print("警告: 找不到中文字体文件，使用默认字体")
        # 尝试使用 Pillow 默认字体
        fontStyle = ImageFont.load_default()
    
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# 自适应帧率控制
def adaptive_frame_skip():
    """根据当前处理性能自适应调整帧跳过数"""
    global PROCESS_EVERY_N_FRAMES
    
    if len(fps_stats) < 10:
        return PROCESS_EVERY_N_FRAMES
    
    avg_fps = sum(fps_stats) / len(fps_stats)
    
    if avg_fps < 15:  # 如果帧率过低
        PROCESS_EVERY_N_FRAMES = min(6, PROCESS_EVERY_N_FRAMES + 1)  # 增加跳帧数，但最多6帧
    elif avg_fps > 25 and PROCESS_EVERY_N_FRAMES > 1:  # 如果帧率足够高
        PROCESS_EVERY_N_FRAMES = max(1, PROCESS_EVERY_N_FRAMES - 1)  # 减少跳帧数，但至少1帧
    
    return PROCESS_EVERY_N_FRAMES

# 获取ToF传感器数据 (如果可用)
def update_tof_distance():
    global tof_distance
    try:
        # 尝试获取ToF传感器数据，如果不可用则默认为0
        tof_distance = tello.get_distance_tof()
    except:
        tof_distance = 0

# 图像分割为5个区域检测障碍物
def detect_obstacles(frame):
    global obstacle_regions
    
    # 优化: 降低分辨率进行障碍物检测
    small_frame = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
    
    # 将画面分为5个区域: 左、上、中、右、下
    height, width = small_frame.shape[:2]
    left_region = small_frame[:, :width//3]
    top_region = small_frame[:height//3, width//3:2*width//3]
    center_region = small_frame[height//3:2*height//3, width//3:2*width//3]
    right_region = small_frame[:, 2*width//3:]
    bottom_region = small_frame[2*height//3:, width//3:2*width//3]
    
    regions = [left_region, top_region, center_region, right_region, bottom_region]
    new_obstacle_regions = [False] * 5
    
    # 使用简单的边缘检测来识别潜在障碍物
    for i, region in enumerate(regions):
        gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # 统计边缘像素数量
        edge_count = np.count_nonzero(edges)
        
        # 根据边缘像素密度判断是否有障碍物
        # 这里的阈值需要根据实际情况调整
        edge_threshold = region.shape[0] * region.shape[1] * 0.03
        new_obstacle_regions[i] = edge_count > edge_threshold
    
    # 平滑障碍物检测结果 (避免闪烁)
    for i in range(5):
        obstacle_regions[i] = obstacle_regions[i] or new_obstacle_regions[i]
    
    return obstacle_regions

# 检测其他人作为避障对象
def detect_other_people(pose_results, target_id, frame):
    other_people = []
    
    if pose_results and len(pose_results) > 1:
        for i, result in enumerate(pose_results):
            if i != target_id and result.pose_landmarks:
                # 获取人体中心点
                landmarks = result.pose_landmarks.landmark
                nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                
                # 转换为像素坐标
                person_x = int(nose.x * frame.shape[1])
                person_y = int(nose.y * frame.shape[0])
                
                # 添加到其他人列表
                other_people.append((person_x, person_y))
                
                # 在图像上标记其他人
                cv2.circle(frame, (person_x, person_y), 10, (255, 165, 0), -1)
                frame = cv2_add_chinese_text(frame, "其他人", (person_x+15, person_y-15), (255, 165, 0), 25)
    
    return other_people, frame

# 基于障碍物和其他人位置计算避障向量
def calculate_avoidance_vector(obstacle_regions, other_people, target_position):
    # 初始化避障向量
    avoid_x, avoid_y = 0, 0
    
    # 处理区域障碍物
    if obstacle_regions[0]:  # 左侧有障碍物
        avoid_x += 20  # 向右避开
    if obstacle_regions[3]:  # 右侧有障碍物
        avoid_x -= 20  # 向左避开
    if obstacle_regions[1]:  # 上方有障碍物
        avoid_y += 20  # 向下避开
    if obstacle_regions[4]:  # 下方有障碍物
        avoid_y -= 20  # 向上避开
    
    # 处理ToF传感器数据
    if tof_distance > 0 and tof_distance < min_safe_tof_distance:
        # ToF检测到前方障碍物，后退
        avoid_x -= 20 * math.cos(math.radians(0))  # 假设ToF传感器朝向前方
        avoid_y -= 20 * math.sin(math.radians(0))
    
    # 处理其他人
    if target_position:
        for person_x, person_y in other_people:
            # 计算与其他人的距离
            dx = target_position[0] - person_x
            dy = target_position[1] - person_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # 如果距离小于安全距离，添加避让向量
            if distance < person_safety_distance:
                # 归一化方向向量
                if distance > 0:
                    norm_dx = dx / distance
                    norm_dy = dy / distance
                else:
                    norm_dx, norm_dy = 1, 0
                
                # 避让力度与距离成反比
                force = (person_safety_distance - distance) / person_safety_distance * 30
                avoid_x += norm_dx * force
                avoid_y += norm_dy * force
    
    return avoid_x, avoid_y
# 创建手部检测线程函数
def hands_detection_thread():
    """在独立线程中执行手部检测"""
    global frame_buffer, hands_results, hands_processing_active
    
    frame_count = 0
    last_process_time = time.time()
    
    # 提高手部检测质量的参数
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,  # 增加信心度阈值
        min_tracking_confidence=0.4,   # 增加跟踪信心度阈值
        model_complexity=1            # 使用更复杂的模型来提高精度
    ) as hands_detector:
        while hands_processing_active:
            # 按照自适应帧率跳过帧
            skip_frames = adaptive_frame_skip()
            
            # 确保帧缓冲区有内容
            if len(frame_buffer) > 0:
                frame_count += 1
                
                # 跳过一些帧以减轻处理负担
                if frame_count % skip_frames != 0:
                    continue
                
                # 获取最新帧
                frame = frame_buffer[-1].copy()
                
                # 使用低分辨率进行处理
                if LOW_RES_PROCESSING:
                    process_frame = cv2.resize(frame, PROCESS_RESOLUTION)
                else:
                    process_frame = frame.copy()
                
                # 转换颜色空间(MediaPipe需要RGB)
                image_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # 进行手部检测
                results = hands_detector.process(image_rgb)
                
                # 更新全局手部检测结果
                hands_results = results
            
            # 避免过高CPU使用率
            time.sleep(0.01)

# 使用MediaPipe Hands检测剪刀手
def detect_peace_sign_with_hands():
    """使用更宽容的标准检测剪刀手手势"""
    global hands_results, peace_sign_detected, peace_sign_cooldown, performing_photo_stunt
    
    if peace_sign_cooldown > 0 or performing_photo_stunt:
        peace_sign_cooldown -= 1
        return False
    
    if not hands_results or not hands_results.multi_hand_landmarks:
        return False
    
    for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
        if hand_idx >= len(hands_results.multi_handedness):
            continue
            
        # Get hand type if available
        is_right_hand = True
        if len(hands_results.multi_handedness) > hand_idx:
            handedness = hands_results.multi_handedness[hand_idx].classification[0].label
            is_right_hand = (handedness == "Right")
        
        # Extract finger landmarks
        index_tip = hand_landmarks.landmark[8]    # 食指指尖
        index_pip = hand_landmarks.landmark[6]    # 食指中间关节
        index_mcp = hand_landmarks.landmark[5]    # 食指根部
        
        middle_tip = hand_landmarks.landmark[12]  # 中指指尖
        middle_pip = hand_landmarks.landmark[10]  # 中指中间关节
        
        ring_tip = hand_landmarks.landmark[16]    # 无名指指尖
        pinky_tip = hand_landmarks.landmark[20]   # 小指指尖
        
        # 1. More flexible check if index and middle fingers are extended
        # Check both vertical and slight angle orientations
        index_extended = (index_tip.y < index_pip.y - 0.01)
        middle_extended = (middle_tip.y < middle_pip.y - 0.01)
        
        # 2. Check if ring and pinky fingers are lower than index/middle
        # This is more reliable than checking if they're bent
        other_fingers_lower = ((ring_tip.y > index_tip.y) and 
                              (pinky_tip.y > index_tip.y))
        
        # 3. Check finger separation with a lower threshold
        finger_distance = math.sqrt(
            (index_tip.x - middle_tip.x)**2 + 
            (index_tip.y - middle_tip.y)**2
        )
        fingers_apart = finger_distance > 0.015  # Lower threshold
        
        # Use more relaxed peace sign detection
        is_peace_sign = index_extended and middle_extended and fingers_apart
        
        # Debug output to help tune parameters
        print(f"Hand analysis: Extended: {index_extended}/{middle_extended}, " 
              f"Others lower: {other_fingers_lower}, Distance: {finger_distance:.4f}")
        
        if is_peace_sign:
            print(f"检测到剪刀手! 指尖距离: {finger_distance:.4f}")
            peace_sign_detected = True
            peace_sign_cooldown = PEACE_SIGN_COOLDOWN
            return True
    
    return False

# 执行拍照和特技动作序列
def perform_photo_stunt_sequence(frame):
    """执行拍照并进行特技飞行"""
    global performing_photo_stunt, stunt_position, pError_x, pError_y, pError_z, pError_yaw, peace_sign_detected
    
    try:
        print("开始执行拍照特技动作")
        performing_photo_stunt = True
        # Record current position
        stunt_position = (pError_x, pError_y, pError_z, pError_yaw)
        
        # 0. 拍照前先移动到更高的位置以更好地拍摄人脸
        try:
            # 稍微上升一点，以便更好地拍摄人脸
            tello.send_rc_control(0, 0, 20, 0)  # 上升
            time.sleep(0.5)
            tello.send_rc_control(0, 0, 0, 0)   # 停止
            time.sleep(0.3)
        except Exception as e:
            print(f"调整高度失败: {e}")
        
        # 1. 获取原始视频帧（无UI叠加）并保存
        try:
            # 获取无人机视频帧（无UI叠加）
            raw_frame = tello.get_frame_read().frame
            # 调整分辨率
            raw_frame = cv2.resize(raw_frame, DISPLAY_RESOLUTION)
            # 转换BGR到RGB然后再到BGR (修复蓝精灵问题)
            raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            # 保存照片
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            photo_path = f"tello_photo_{timestamp}.jpg"
            cv2.imwrite(photo_path, raw_frame)
            print(f"照片已保存: {photo_path}")
        except Exception as e:
            print(f"拍照失败: {e}")
            # 如果无法获取原始帧，则使用当前帧（可能包含UI）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            photo_path = f"tello_photo_{timestamp}.jpg"
            cv2.imwrite(photo_path, frame)
            print(f"使用处理后的帧保存照片: {photo_path}")
        
        
        # 3. 执行后翻转 - 添加防失速预处理
        try:
            # 预先加速马达到最高速度而不移动 (短暂快速地左右摇摆)
            print("预热马达到最高转速...")
            tello.send_rc_control(0, 0, 0, 50)  # 快速右转
            time.sleep(0.2)
            tello.send_rc_control(0, 0, 0, -50)  # 快速左转
            time.sleep(0.2)
            tello.send_rc_control(0, 0, 0, 0)    # 停止旋转
            time.sleep(0.1)
            
            # 短暂上升一点以获得更多高度空间
            tello.send_rc_control(0, 0, 40, 0)   # 快速上升
            time.sleep(0.3)
            tello.send_rc_control(0, 0, 0, 0)    # 停止上升
            time.sleep(0.1)
            
            # 现在执行翻转 - 马达已经在高速运行
            print("执行后翻转")
            tello.flip_back()
            time.sleep(2)  # 给足够时间完成翻转
        except Exception as e:
            print(f"翻转失败: {e}")
        # 尝试备用翻转方式
        try:
            print("尝试备用翻转方式")
            # 再次预热马达
            tello.send_rc_control(0, 0, 30, 0)  # 上升
            time.sleep(0.3)
            tello.send_rc_control(0, 0, 0, 0)   # 停止
            time.sleep(0.1)
            
            # 使用直接命令
            tello.send_command_without_return("flip b")
            time.sleep(2)
        except Exception as e2:
            print(f"备用翻转也失败: {e2}")
                
        # 4. 保持马达旋转 - 通过快速左右小幅度旋转实现
        try:
            for _ in range(3):  # 重复几次小幅度旋转
                tello.send_rc_control(0, 0, 0, 30)  # 右转
                time.sleep(0.3)
                tello.send_rc_control(0, 0, 0, -30)  # 左转
                time.sleep(0.3)
            tello.send_rc_control(0, 0, 0, 0)  # 停止旋转
            time.sleep(0.5)
        except Exception as e:
            print(f"马达旋转失败: {e}")
        
        # 5. 恢复到原始位置
        try:
            # 使用原来的误差参数来恢复相对位置
            error_x, error_y, error_z, error_yaw = stunt_position
            
            # 计算PID控制
            speed_x = pid_x[0] * error_x
            speed_y = pid_y[0] * error_y
            speed_z = pid_z[0] * error_z
            speed_yaw = pid_yaw[0] * error_yaw
            
            # 限制速度范围
            lr_velocity = int(np.clip(speed_x, -20, 20))
            fb_velocity = int(np.clip(speed_z, -20, 20))
            ud_velocity = int(-np.clip(speed_y, -20, 20))
            yaw_velocity = int(np.clip(speed_yaw, -20, 20))
            
            # 发送控制命令，恢复位置
            tello.send_rc_control(lr_velocity, fb_velocity, ud_velocity, yaw_velocity)
            time.sleep(1.5)
            
            # 恢复悬停
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"位置恢复失败: {e}")
    except Exception as e:
        print(f"拍照特技执行错误: {e}")
    
    performing_photo_stunt = False
# 人体目标选择函数
def select_target(results_list, predicted_position=None):
    """从多个检测结果中选择目标，考虑预测位置"""
    if not results_list:
        return None
    
    # 获取帧尺寸
    frame_width = DISPLAY_RESOLUTION[0]
    frame_height = DISPLAY_RESOLUTION[1]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    # 如果有预测位置，倾向于选择接近预测位置的人
    if predicted_position is not None and prediction_enabled:
        closest_dist = float('inf')
        closest_idx = -1
        
        for i, result in enumerate(results_list):
            landmarks = result.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nose_x = int(nose.x * frame_width)
            nose_y = int(nose.y * frame_height)
            
            # 计算到预测位置的距离
            dist = math.sqrt((nose_x - predicted_position[0])**2 + (nose_y - predicted_position[1])**2)
            
            if dist < closest_dist:
                closest_dist = dist
                closest_idx = i
        
        # 如果找到的目标与预测位置足够接近，返回该目标
        if closest_dist < frame_width * 0.4:  # 预测位置误差阈值
            return closest_idx
    
    # 如果没有可信的预测位置或预测失败，选择距离画面中心最近的人
    closest_dist = float('inf')
    closest_idx = -1
    
    for i, result in enumerate(results_list):
        landmarks = result.pose_landmarks.landmark
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        nose_x = int(nose.x * frame_width)
        nose_y = int(nose.y * frame_height)
        dist = math.sqrt((nose_x - frame_center_x)**2 + (nose_y - frame_center_y)**2)
        
        if dist < closest_dist:
            closest_dist = dist
            closest_idx = i
    
    return closest_idx

# 预测未来位置
def predict_position(history, velocities, time_steps=1):
    """基于历史位置和速度预测未来位置"""
    if len(history) < 3:
        return None
    
    # 获取帧尺寸
    frame_width = DISPLAY_RESOLUTION[0]
    frame_height = DISPLAY_RESOLUTION[1]
    
    # 最近的位置
    recent_positions = list(history)[-3:]
    
    # 计算最近的速度向量 (移动方向和幅度)
    dx = recent_positions[-1][0] - recent_positions[0][0]
    dy = recent_positions[-1][1] - recent_positions[0][1]
    
    # 计算平滑的平均速度
    if len(velocities) > 0:
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        # 结合当前速度和历史平均速度
        vx = 0.6 * dx / 2 + 0.4 * avg_vx  # 60%当前速度, 40%历史平均
        vy = 0.6 * dy / 2 + 0.4 * avg_vy
    else:
        vx = dx / 2
        vy = dy / 2
    
    # 存储当前速度
    if abs(vx) > 0.5 or abs(vy) > 0.5:  # 仅存储有意义的速度
        velocities.append((vx, vy))
    
    # 预测未来位置 (当前位置 + 速度 * 时间步长)
    future_x = int(recent_positions[-1][0] + vx * time_steps * 1.2)  # 稍微增大预测幅度
    future_y = int(recent_positions[-1][1] + vy * time_steps * 1.2)
    
    # 限制在画面范围内
    future_x = max(0, min(frame_width, future_x))
    future_y = max(0, min(frame_height, future_y))
    
    return (future_x, future_y)

# 发送控制命令
def send_control_command(lr, fb, ud, yaw):
    """发送控制命令，确保命令值不会太小"""
    global last_command_time, keepalive_counter
    
    # 确保控制值不会太小，避免自动降落
    if 0 < abs(lr) < MIN_CONTROL_VALUE:
        lr = MIN_CONTROL_VALUE if lr > 0 else -MIN_CONTROL_VALUE
    if 0 < abs(fb) < MIN_CONTROL_VALUE:
        fb = MIN_CONTROL_VALUE if fb > 0 else -MIN_CONTROL_VALUE
    if 0 < abs(ud) < MIN_CONTROL_VALUE:
        ud = MIN_CONTROL_VALUE if ud > 0 else -MIN_CONTROL_VALUE
    if 0 < abs(yaw) < MIN_CONTROL_VALUE:
        yaw = MIN_CONTROL_VALUE if yaw > 0 else -MIN_CONTROL_VALUE
    
    # 如果所有控制值都接近0，发送微小的控制指令保持连接
    if abs(lr) < 2 and abs(fb) < 2 and abs(ud) < 2 and abs(yaw) < 2:
        keepalive_counter = (keepalive_counter + 1) % 4
        if keepalive_counter == 0:
            yaw = 2
        elif keepalive_counter == 2:
            yaw = -2
    
    # 发送控制命令
    tello.send_rc_control(int(lr), int(fb), int(ud), int(yaw))
    last_command_time = time.time()

# 执行特技动作 - 改进版眼镜蛇机动
def perform_stunt_sequence():
    global performing_stunt, stunt_position, pError_x, pError_y, pError_z, pError_yaw
    
    # 记录当前高度和位置
    try:
        current_height = tello.get_height()
        stunt_position = (pError_x, pError_y, pError_z, pError_yaw)
        
        # 确保高度充足 (特技动作需要一定高度)
        if current_height < 100:  # 如果低于100厘米，先升高
            tello.move_up(100 - current_height)
            time.sleep(2)
        
        # 执行向后翻转
        try:
            tello.flip_back()  # 使用back翻转
            time.sleep(3)  # 给足够的时间完成翻转并稳定
        except Exception as e:
            print(f"翻转失败: {e}，执行备用特技")
            # 备用特技：快速上升后下降
            tello.move_up(30)
            time.sleep(1)
            tello.move_down(30)
            time.sleep(1)
        
        # 眼镜蛇机动 (更精确的战斗机动作模拟)
        try:
            # 1. 快速上升 (机头抬起)
            tello.send_rc_control(0, 0, 50, 0)  # 快速向上
            time.sleep(0.8)
            tello.send_rc_control(0, 0, 0, 0)   # 停止
            time.sleep(0.2)
            
            # 2. 短暂保持高仰角 (几乎垂直) + 减速
            tello.send_rc_control(0, -20, 30, 0)  # 轻微后退+继续上升
            time.sleep(0.5)
            tello.send_rc_control(0, 0, 0, 0)   # 停止
            time.sleep(0.3)
            
            # 3. 迅速恢复水平姿态并加速前进
            tello.send_rc_control(0, 50, -20, 0)  # 快速前进并下降
            time.sleep(0.8)
            tello.send_rc_control(0, 30, 0, 0)   # 继续前进
            time.sleep(0.5)
            tello.send_rc_control(0, 0, 0, 0)    # 停止
            time.sleep(0.5)
        except Exception as e:
            print(f"眼镜蛇机动失败: {e}")
        
        # 恢复到原始位置
        try:
            # 使用原来的误差参数来恢复相对位置
            error_x, error_y, error_z, error_yaw = stunt_position
            
            # 计算PID控制
            speed_x = pid_x[0] * error_x
            speed_y = pid_y[0] * error_y
            speed_z = pid_z[0] * error_z
            speed_yaw = pid_yaw[0] * error_yaw
            
            # 限制速度范围
            lr_velocity = int(np.clip(speed_x, -20, 20))
            fb_velocity = int(np.clip(speed_z, -20, 20))
            ud_velocity = int(-np.clip(speed_y, -20, 20))
            yaw_velocity = int(np.clip(speed_yaw, -20, 20))
            
            # 发送控制命令，恢复位置
            tello.send_rc_control(lr_velocity, fb_velocity, ud_velocity, yaw_velocity)
            time.sleep(2)
            
            # 恢复悬停
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(0.5)
        except Exception as e:
            print(f"位置恢复失败: {e}")
    except Exception as e:
        print(f"特技执行错误: {e}")
    
    performing_stunt = False

# 检测右手挥手动作
def detect_right_hand_wave(landmarks):
    """检测右手挥手动作"""
    global right_wrist_history, right_wave_count, right_wave_direction, right_wave_detected, right_wave_cooldown
    
    # 获取帧尺寸
    frame_width = DISPLAY_RESOLUTION[0]
    frame_height = DISPLAY_RESOLUTION[1]
    
    # 如果冷却时间未到，不进行检测
    if right_wave_cooldown > 0:
        right_wave_cooldown -= 1
        return False
    
    # 获取右手腕和右肘关键点
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # 获取左手腕关键点 (检查左手是否也在移动)
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    
    # 检查关键点的可见性
    if (right_wrist.visibility < 0.5 or right_elbow.visibility < 0.5 or 
        right_shoulder.visibility < 0.5 or left_wrist.visibility < 0.5 or 
        left_elbow.visibility < 0.5):
        # 关键点不可见，重置状态
        right_wrist_history.clear()
        right_wave_count = 0
        right_wave_direction = 0
        return False
    
    # 转换为像素坐标
    right_wrist_x = int(right_wrist.x * frame_width)
    right_wrist_y = int(right_wrist.y * frame_height)
    right_elbow_x = int(right_elbow.x * frame_width)
    right_elbow_y = int(right_elbow.y * frame_height)
    right_shoulder_x = int(right_shoulder.x * frame_width)
    right_shoulder_y = int(right_shoulder.y * frame_height)
    
    left_wrist_x = int(left_wrist.x * frame_width)
    left_wrist_y = int(left_wrist.y * frame_height)
    left_elbow_x = int(left_elbow.x * frame_width)
    left_elbow_y = int(left_elbow.y * frame_height)
    
    # 存储右手腕位置、右肘位置和左手腕位置
    right_wrist_history.append((
        (right_wrist_x, right_wrist_y), 
        (right_elbow_x, right_elbow_y),
        (left_wrist_x, left_wrist_y)
    ))
    
    # 至少需要2帧才能检测移动
    if len(right_wrist_history) < 2:
        return False
    
    # 获取当前和前一帧的右手腕位置
    curr_wrist, curr_elbow, curr_left_wrist = right_wrist_history[-1]
    prev_wrist, prev_elbow, prev_left_wrist = right_wrist_history[-2]
    
    # 检查左手是否在移动
    left_wrist_movement = math.sqrt((curr_left_wrist[0] - prev_left_wrist[0])**2 + 
                                  (curr_left_wrist[1] - prev_left_wrist[1])**2)
    left_wrist_moving = left_wrist_movement > wave_threshold
    
    # 计算右手腕相对于右肘的水平移动
    # 使用相对移动可以减少身体整体移动的影响
    relative_curr_x = curr_wrist[0] - curr_elbow[0]
    relative_prev_x = prev_wrist[0] - prev_elbow[0]
    horizontal_movement = relative_curr_x - relative_prev_x
    
    # 检查右手是否高于肩膀 (通常挥手时手会抬高)
    hand_raised = curr_wrist[1] < right_shoulder_y
    
    # 检查右手臂是否伸展 (手腕和肘部有一定距离)
    arm_extended = math.sqrt((curr_wrist[0] - curr_elbow[0])**2 + 
                            (curr_wrist[1] - curr_elbow[1])**2) > 50
    
    # 如果左手在移动，不认为是单纯的右手挥手
    if left_wrist_moving:
        right_wave_count = 0
        right_wave_direction = 0
        return False
    
    # 检测挥手动作的方向变化
    if abs(horizontal_movement) > wave_threshold and hand_raised and arm_extended:
        current_direction = 1 if horizontal_movement > 0 else -1
        
        # 当检测到方向改变且与前一个方向相反时，计为一次挥手
        if right_wave_direction != 0 and current_direction != right_wave_direction:
            right_wave_count += 1
            
        right_wave_direction = current_direction
    
    # 判断是否完成了足够次数的挥手动作
    if right_wave_count >= REQUIRED_WAVES:
        right_wave_detected = True
        right_wave_count = 0
        right_wave_direction = 0
        right_wave_cooldown = 30  # 设置冷却时间，防止连续误触
        right_wrist_history.clear()
        return True
    
    return False

# 检测左手挥手动作
def detect_left_hand_wave(landmarks):
    """检测左手挥手动作"""
    global left_wrist_history, left_wave_count, left_wave_direction, left_wave_detected, left_wave_cooldown, performing_stunt
    
    # 获取帧尺寸
    frame_width = DISPLAY_RESOLUTION[0]
    frame_height = DISPLAY_RESOLUTION[1]
    
    # 如果冷却时间未到或正在执行特技，不进行检测
    if left_wave_cooldown > 0 or performing_stunt:
        left_wave_cooldown -= 1
        return False
    
    # 获取左手腕和左肘关键点
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    
    # 获取右手腕关键点 (检查右手是否也在移动)
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    
    # 检查关键点的可见性
    if (left_wrist.visibility < 0.5 or left_elbow.visibility < 0.5 or 
        left_shoulder.visibility < 0.5 or right_wrist.visibility < 0.5 or 
        right_elbow.visibility < 0.5):
        # 关键点不可见，重置状态
        left_wrist_history.clear()
        left_wave_count = 0
        left_wave_direction = 0
        return False
    
    # 转换为像素坐标
    left_wrist_x = int(left_wrist.x * frame_width)
    left_wrist_y = int(left_wrist.y * frame_height)
    left_elbow_x = int(left_elbow.x * frame_width)
    left_elbow_y = int(left_elbow.y * frame_height)
    left_shoulder_x = int(left_shoulder.x * frame_width)
    left_shoulder_y = int(left_shoulder.y * frame_height)
    
    right_wrist_x = int(right_wrist.x * frame_width)
    right_wrist_y = int(right_wrist.y * frame_height)
    right_elbow_x = int(right_elbow.x * frame_width)
    right_elbow_y = int(right_elbow.y * frame_height)
    
    # 存储左手腕位置、左肘位置和右手腕位置
    left_wrist_history.append((
        (left_wrist_x, left_wrist_y), 
        (left_elbow_x, left_elbow_y),
        (right_wrist_x, right_wrist_y)
    ))
    
    # 至少需要2帧才能检测移动
    if len(left_wrist_history) < 2:
        return False
    
    # 获取当前和前一帧的左手腕位置
    curr_wrist, curr_elbow, curr_right_wrist = left_wrist_history[-1]
    prev_wrist, prev_elbow, prev_right_wrist = left_wrist_history[-2]
    
    # 检查右手是否在移动
    right_wrist_movement = math.sqrt((curr_right_wrist[0] - prev_right_wrist[0])**2 + 
                                   (curr_right_wrist[1] - prev_right_wrist[1])**2)
    right_wrist_moving = right_wrist_movement > wave_threshold
    
    # 计算左手腕相对于左肘的水平移动
    relative_curr_x = curr_wrist[0] - curr_elbow[0]
    relative_prev_x = prev_wrist[0] - prev_elbow[0]
    horizontal_movement = relative_curr_x - relative_prev_x
    
    # 检查左手是否高于肩膀
    hand_raised = curr_wrist[1] < left_shoulder_y
    
    # 检查左手臂是否伸展
    arm_extended = math.sqrt((curr_wrist[0] - curr_elbow[0])**2 + 
                            (curr_wrist[1] - curr_elbow[1])**2) > 50
    
    # 如果右手在移动，不认为是单纯的左手挥手
    if right_wrist_moving:
        left_wave_count = 0
        left_wave_direction = 0
        return False
    
    # 检测挥手动作的方向变化
    if abs(horizontal_movement) > wave_threshold and hand_raised and arm_extended:
        current_direction = 1 if horizontal_movement > 0 else -1
        
        # 当检测到方向改变且与前一个方向相反时，计为一次挥手
        if left_wave_direction != 0 and current_direction != left_wave_direction:
            left_wave_count += 1
            
        left_wave_direction = current_direction
    
    # 判断是否完成了足够次数的挥手动作
    if left_wave_count >= REQUIRED_WAVES:
        left_wave_detected = True
        left_wave_count = 0
        left_wave_direction = 0
        left_wave_cooldown = 30  # 设置冷却时间，防止连续误触
        left_wrist_history.clear()
        return True
    
    return False

# 人体跟随函数
def track_person(frame, results, pid_x, pid_y, pid_z, pid_yaw, pError_x, pError_y, pError_z, pError_yaw):
    global position_history, velocity_history, last_movement_direction, target_id
    global target_position, target_lost_frames, right_wave_detected, left_wave_detected
    global performing_stunt, stunt_position, latest_landmarks
    
    # 获取帧尺寸
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2
    
    # 更新全局landmarks (用于其他线程访问)
    if results and results.pose_landmarks:
        latest_landmarks = results.pose_landmarks
    
    # 如果正在执行特技，不进行跟踪
    if performing_stunt:
        frame = cv2_add_chinese_text(frame, "正在执行特技动作...", (frame_center_x-120, frame_center_y), (0, 165, 255), 40)
        return frame, pError_x, pError_y, pError_z, pError_yaw
    
    # 确保results是一个列表
    if not isinstance(results, list):
        pose_results = [results] if results and results.pose_landmarks else []
    else:
        pose_results = [r for r in results if r.pose_landmarks]
    
    # 预测位置 (基于历史位置)
    predicted_position = None
    if target_position is not None and len(position_history) >= 3:
        predicted_position = predict_position(position_history, velocity_history)
        
        # 显示预测位置
        if predicted_position:
            cv2.circle(frame, predicted_position, 10, (0, 165, 255), -1)  # 橙色圆点
            cv2.putText(frame, "Predicted", (predicted_position[0] + 15, predicted_position[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
    
    # 如果没有检测到人体
    if not pose_results:
        target_lost_frames += 1
        
        # 如果有预测位置且丢失帧数不多，基于预测位置控制
        if predicted_position and target_lost_frames < MAX_LOST_FRAMES//2 and prediction_enabled:
            # 计算到预测位置的误差
            error_x = predicted_position[0] - frame_center_x
            error_y = predicted_position[1] - frame_center_y
            
            # 基于预测位置的PID控制
            speed_x = pid_x[0] * error_x * 0.8  # 降低预测控制的响应度
            speed_y = pid_y[0] * error_y * 0.8
            
            # 保持当前yaw和z方向
            yaw_velocity = pError_yaw * 0.7  # 平滑降低旋转速度
            
            # 发送预测控制命令
            send_control_command(
                int(np.clip(speed_x, -20, 20)),
                0,  # 不做前后移动
                int(-np.clip(speed_y, -20, 20)),
                int(np.clip(yaw_velocity, -20, 20))
            )
            
            # 在框架上显示"基于预测移动"
            frame = cv2_add_chinese_text(frame, "基于预测移动", (frame_width//2-80, 30), (0, 165, 255), 30)
        else:
            # 超过预测阈值，发送保持活动的命令
            if target_lost_frames > MAX_LOST_FRAMES:
                target_id = None
                target_position = None
                
            send_control_command(0, 0, 0, 2 if keepalive_counter % 2 == 0 else -2)
            
            # 在框架上添加"没有检测到人体"
            frame = cv2_add_chinese_text(frame, "没有检测到人体", (10, 30), (0, 0, 255), 30)
        
        return frame, pError_x, pError_y, pError_z, pError_yaw
    
    # 检测障碍物
    if obstacle_detection_enabled:
        # 更新ToF传感器数据
        update_tof_distance()
        
        # 检测图像中的障碍物
        obstacle_regions = detect_obstacles(frame)
        
        # 在图像上显示障碍物区域
        regions_names = ["左", "上", "中", "右", "下"]
        for i, has_obstacle in enumerate(obstacle_regions):
            color = (0, 0, 255) if has_obstacle else (0, 255, 0)
            frame = cv2_add_chinese_text(frame, f"{regions_names[i]}: {'有' if has_obstacle else '无'}", 
                                       (frame_width-120, 150+30*i), color, 25)
    
    # 如果尚未选择目标或目标已丢失太久，选择新目标
    if target_id is None or target_lost_frames > MAX_LOST_FRAMES:
        target_id = select_target(pose_results, predicted_position)
        target_lost_frames = 0
        position_history.clear()  # 清空历史位置
        velocity_history.clear()  # 清空速度历史
    
    # 如果有多个人，且已经有目标，保持跟踪相同的人
    elif len(pose_results) > 1 and target_position is not None:
        # 寻找与上一帧位置或预测位置最接近的人
        min_dist = float('inf')
        closest_id = target_id
        
        reference_pos = predicted_position if predicted_position and prediction_enabled else target_position
        
        for i, result in enumerate(pose_results):
            landmarks = result.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            nose_x = int(nose.x * frame_width)
            nose_y = int(nose.y * frame_height)
            
            dist = math.sqrt((nose_x - reference_pos[0])**2 + (nose_y - reference_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_id = i
        
        target_id = closest_id
    
    # 确保目标ID有效
    target_id = 0 if target_id is None or target_id >= len(pose_results) else target_id
    
    # 获取目标人体的关键点
    landmarks = pose_results[target_id].pose_landmarks.landmark
    
    # 检测右手挥手动作
    if detect_right_hand_wave(landmarks):
        right_wave_detected = True
        frame = cv2_add_chinese_text(frame, "检测到右手挥手! 准备降落...", 
                                    (frame_center_x-150, frame_center_y), (0, 0, 255), 40)
        return frame, pError_x, pError_y, pError_z, pError_yaw
    
    # 检测左手挥手动作
    if detect_left_hand_wave(landmarks):
        left_wave_detected = True
        frame = cv2_add_chinese_text(frame, "检测到左手挥手! 准备执行特技...", 
                                   (frame_center_x-150, frame_center_y), (0, 165, 255), 40)
        performing_stunt = True
        # 在新线程中执行特技动作，避免阻塞主线程
        threading.Thread(target=perform_stunt_sequence).start()
        return frame, pError_x, pError_y, pError_z, pError_yaw
        
    # 检测剪刀手手势(使用MediaPipe Hands)
    if detect_peace_sign_with_hands():
        peace_sign_detected = True
        frame = cv2_add_chinese_text(frame, "检测到剪刀手! 准备拍照并执行特技...", 
                                   (frame_center_x-180, frame_center_y), (0, 255, 255), 40)
                                   
        # 在新线程中执行拍照和特技动作，避免阻塞主线程
        photo_thread = threading.Thread(target=lambda: perform_photo_stunt_sequence(frame.copy()))
        photo_thread.daemon = True
        photo_thread.start()
        return frame, pError_x, pError_y, pError_z, pError_yaw
    
    # 绘制右手轨迹用于调试
    if len(right_wrist_history) >= 2:
        for i in range(1, len(right_wrist_history)):
            prev_pos = right_wrist_history[i-1][0]  # 右手腕位置
            curr_pos = right_wrist_history[i][0]
            # 绘制右手腕轨迹
            cv2.line(frame, prev_pos, curr_pos, (255, 0, 0), 2)
    
    # 绘制左手轨迹用于调试
    if len(left_wrist_history) >= 2:
        for i in range(1, len(left_wrist_history)):
            prev_pos = left_wrist_history[i-1][0]  # 左手腕位置
            curr_pos = left_wrist_history[i][0]
            # 绘制左手腕轨迹
            cv2.line(frame, prev_pos, curr_pos, (0, 0, 255), 2)
    
    # 显示挥手检测状态
    frame = cv2_add_chinese_text(frame, f"右手挥手: {right_wave_count}/{REQUIRED_WAVES}", 
                (10, frame_height - 150), (255, 0, 0), 25)
    frame = cv2_add_chinese_text(frame, f"左手挥手: {left_wave_count}/{REQUIRED_WAVES}", 
                (10, frame_height - 180), (0, 0, 255), 25)
    
    # 使用躯干中心点
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    # 计算躯干中心
    torso_x = (left_hip.x + right_hip.x + left_shoulder.x + right_shoulder.x) / 4
    torso_y = (left_hip.y + right_hip.y + left_shoulder.y + right_shoulder.y) / 4
    
    # 转换为像素坐标
    center_x = int(torso_x * frame_width)
    center_y = int(torso_y * frame_height)
    
    # 更新目标位置
    new_target_position = (center_x, center_y)
    
    # 计算人体移动方向和幅度
    horizontal_movement = 0
    if target_position:
        horizontal_movement = new_target_position[0] - target_position[0]
        
        # 显示移动方向和大小
        movement_text = f"横向移动: {horizontal_movement}"
        frame = cv2_add_chinese_text(frame, movement_text, (10, frame_height - 60), (0, 255, 0), 30)
    
    target_position = new_target_position
    target_lost_frames = 0
    
    # 在目标人体中心绘制特殊标记
    cv2.circle(frame, target_position, 10, (0, 255, 0), -1)
    
    # 添加目标标记
    frame = cv2_add_chinese_text(frame, "目标", (target_position[0] + 20, target_position[1] - 20), (0, 255, 0), 30)
    
    # 保存当前位置到历史记录
    position_history.append(target_position)
    
    # 检测和标记其他人，用于避障
    other_people = []
    if len(pose_results) > 1:
        other_people, frame = detect_other_people(pose_results, target_id, frame)
    
    # 计算避障向量
    avoid_x, avoid_y = 0, 0
    if obstacle_detection_enabled:
        avoid_x, avoid_y = calculate_avoidance_vector(obstacle_regions, other_people, target_position)
    
    # 计算人体中心与画面中心的误差
    error_x = target_position[0] - frame_center_x
    error_y = target_position[1] - frame_center_y
    
    # 计算人体相对于画面中心的角度 (用于旋转控制)
    angle_to_center = math.atan2(error_x, frame_width/3) * 180 / math.pi
    error_yaw = angle_to_center
    
    # 估计距离 (基于肩膀宽度)
    shoulder_width = abs(left_shoulder.x - right_shoulder.x) * frame_width
    ideal_width = frame_width * 0.2
    error_z = ideal_width - shoulder_width
    
    # 检测人体运动方向 (如果有足够的历史数据)
    if len(position_history) >= 5:
        old_pos = position_history[0]
        current_pos = position_history[-1]
        
        # x方向移动
        movement_x = current_pos[0] - old_pos[0]
        
        # 使用nose.z估计前后移动
        if nose.z < 0:
            movement_z = 1  # 前进
        else:
            movement_z = -1  # 后退
            
        # 平滑移动方向
        last_movement_direction[0] = 0.7 * last_movement_direction[0] + 0.3 * movement_x
        last_movement_direction[1] = 0.7 * last_movement_direction[1] + 0.3 * movement_z
        
        # 显示移动方向
        direction_text = ""
        if abs(last_movement_direction[0]) > 5:
            direction_text += "左移动 " if last_movement_direction[0] < 0 else "右移动 "
        if abs(last_movement_direction[1]) > 0.2:
            direction_text += "后退" if last_movement_direction[1] < 0 else "前进"
            
        frame = cv2_add_chinese_text(frame, f"移动方向: {direction_text}", (10, frame_height - 30), (0, 255, 0), 30)
    
    # PID控制计算
    # X轴控制（左右）
    speed_x = pid_x[0] * error_x + pid_x[1] * (error_x - pError_x) + pid_x[2] * sum([error_x, pError_x])
    speed_x = int(np.clip(speed_x, -30, 30))
    
    # Y轴控制（上下）
    speed_y = pid_y[0] * error_y + pid_y[1] * (error_y - pError_y) + pid_y[2] * sum([error_y, pError_y])
    speed_y = int(np.clip(speed_y, -30, 30))
    
    # Z轴控制（前后）
    speed_z = pid_z[0] * error_z + pid_z[1] * (error_z - pError_z) + pid_z[2] * sum([error_z, pError_z])
    
    # 前后移动控制
    if len(position_history) >= 5 and last_movement_direction[1] > 0.3:
        speed_z += 15  # 人向前走，无人机也前进
    elif len(position_history) >= 5 and last_movement_direction[1] < -0.3:
        speed_z -= 15  # 人向后走，无人机也后退
        
    speed_z = int(np.clip(speed_z, -30, 30))
    
    # Yaw控制（旋转）
    # 1. 基于人体位置的旋转控制
    position_yaw = pid_yaw[0] * error_yaw + pid_yaw[1] * (error_yaw - pError_yaw) + pid_yaw[2] * sum([error_yaw, pError_yaw])
    
    # 2. 基于人体横向移动的额外旋转控制
    movement_yaw = 0
    if horizontal_movement != 0 and len(position_history) > 2:
        # 当人向左移动(负值)，无人机应该逆时针旋转(负值)
        # 当人向右移动(正值)，无人机应该顺时针旋转(正值)
        movement_yaw = horizontal_movement * 0.5
    
    # 合并两种旋转控制
    speed_yaw = position_yaw + movement_yaw
    speed_yaw = int(np.clip(speed_yaw, -50, 50))
    
    # 在画面上显示旋转控制信息
    frame = cv2_add_chinese_text(frame, f"位置旋转: {int(position_yaw)}", (10, frame_height - 90), (0, 255, 0), 30)
    frame = cv2_add_chinese_text(frame, f"移动旋转: {int(movement_yaw)}", (10, frame_height - 120), (0, 255, 0), 30)
    
    # 使用调整后的旋转与移动比例
    yaw_ratio = min(1.0, abs(error_x) / 100) * ROTATION_PREFERENCE
    
    # 分配控制比例
    adj_speed_x = speed_x * (1 - yaw_ratio)
    adj_speed_yaw = speed_yaw * yaw_ratio
    
    # 添加避障矢量
    if obstacle_detection_enabled and (abs(avoid_x) > 0 or abs(avoid_y) > 0):
        # 显示避障信息
        frame = cv2_add_chinese_text(frame, f"避障: X={avoid_x:.1f}, Y={avoid_y:.1f}", 
                                   (frame_width-200, 300), (0, 0, 255), 25)
        
        # 将避障向量合并到控制中
        adj_speed_x += avoid_x
        speed_y += avoid_y
    
    # 发送控制命令到无人机
    lr_velocity = int(adj_speed_x)
    fb_velocity = speed_z
    ud_velocity = -speed_y
    yaw_velocity = int(speed_yaw)
    
    # 添加死区
    if abs(error_x) <= 30:
        lr_velocity = 0
    if abs(error_y) <= 40:
        ud_velocity = 0
    if abs(error_z) <= 0.05 and abs(last_movement_direction[1]) <= 0.3:
        fb_velocity = 0
    if abs(error_yaw) <= 5 and abs(horizontal_movement) <= 3:
        yaw_velocity = 0
    
    # 使用send_control_command代替tello.send_rc_control
    send_control_command(lr_velocity, fb_velocity, ud_velocity, yaw_velocity)
    
    # 在画面上显示控制数据
    info = [
        f"目标锁定: {'是' if target_id is not None else '否'}",
        f"错误 X: {error_x}",
        f"错误 Y: {error_y}",
        f"错误 Z: {error_z:.2f}",
        f"错误 Yaw: {error_yaw:.1f}°",
        f"速度 左右: {lr_velocity}",
        f"速度 前后: {fb_velocity}",
        f"速度 上下: {ud_velocity}",
        f"速度 旋转: {yaw_velocity}",
        f"ToF: {tof_distance} cm"
    ]
    
    # 使用中文文本函数显示所有信息
    for i, text in enumerate(info):
        frame = cv2_add_chinese_text(frame, text, (10, 30 + i * 30), (0, 255, 0), 25)
    
    return frame, error_x, error_y, error_z, error_yaw

# 视频捕获线程函数
def video_capture_thread():
    """持续从Tello获取视频帧并放入缓冲区"""
    global frame_buffer, processing_active
    
    while processing_active:
        try:
            # 获取无人机视频帧
            frame = tello.get_frame_read().frame
            if frame is not None:
                # 调整分辨率
                resized_frame = cv2.resize(frame, DISPLAY_RESOLUTION)
                
                # 添加到帧缓冲区
                frame_buffer.append(resized_frame)
                
                # 控制帧率
                time.sleep(1/90)  # 尝试以更高的速率捕获帧
        except Exception as e:
            print(f"视频捕获错误: {e}")
            time.sleep(0.1)

# 姿态检测线程函数
def pose_detection_thread():
    """在独立线程中执行姿态检测"""
    global frame_buffer, processed_frames, processing_active, pError_x, pError_y, pError_z, pError_yaw
    
    frame_count = 0
    last_process_time = time.time()
    
    with mp_pose.Pose(
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
        enable_segmentation=True
    ) as pose_detector:
        while processing_active:
            # 按照自适应帧率跳过帧
            skip_frames = adaptive_frame_skip()
            
            # 确保帧缓冲区有内容
            if len(frame_buffer) > 0:
                frame_count += 1
                
                # 跳过一些帧以减轻处理负担
                if frame_count % skip_frames != 0:
                    continue
                
                # 获取最新帧
                frame = frame_buffer[-1].copy()
                
                # 处理性能：使用低分辨率进行处理
                if LOW_RES_PROCESSING:
                    process_frame = cv2.resize(frame, PROCESS_RESOLUTION)
                else:
                    process_frame = frame.copy()
                
                process_start = time.time()
                
                # 转换颜色空间(MediaPipe需要RGB)
                image_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                # 进行人体姿态检测
                results = pose_detector.process(image_rgb)
                
                # 计算处理时间
                process_time = time.time() - process_start
                fps = 1.0 / (time.time() - last_process_time)
                last_process_time = time.time()
                
                # 存储FPS统计
                fps_stats.append(fps)
                
                # 将处理后的结果转换回原始分辨率
                if LOW_RES_PROCESSING and results and results.pose_landmarks:
                    # 调整关键点坐标到显示分辨率
                    for landmark in results.pose_landmarks.landmark:
                        landmark.x = landmark.x
                        landmark.y = landmark.y
                
                # 将RGB图像转回BGR，并处理姿态数据
                display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 绘制所有检测到的人体骨骼
                if results and results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        display_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(66, 245, 117), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(230, 66, 245), thickness=2, circle_radius=1)
                    )
                if hands_results and hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
        # 绘制所有21个手部关键点和连接线
                        mp_drawing.draw_landmarks(
                        display_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
                    )    
                
                # 人体跟随处理
                display_image, px, py, pz, pyaw = track_person(
                    display_image, results, pid_x, pid_y, pid_z, pid_yaw, 
                    pError_x, pError_y, pError_z, pError_yaw
                )
                
                # 更新PID误差值
                pError_x, pError_y, pError_z, pError_yaw = px, py, pz, pyaw
                
                # 放入处理后的帧队列
                if not processed_frames.full():
                    processed_frames.put((display_image, time.time()))
            
            # 避免过高CPU使用率
            time.sleep(0.001)

# 显示线程函数 - 替换了帧插值线程
def display_thread():
    """直接显示处理后的帧，不进行插值"""
    global processed_frames, display_active
    
    while display_active:
        # 如果有处理后的帧
        if not processed_frames.empty():
            frame, _ = processed_frames.get()
            
            # 显示帧
            cv2.imshow('Tello人体跟随', frame)
            
            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if process_key_press(key):
                display_active = False
                return
        
        # 避免过高CPU使用率
        time.sleep(0.001)


# 处理按键事件
def process_key_press(key):
    """处理按键事件，返回True表示退出程序"""
    global target_id, target_position, position_history, velocity_history
    global prediction_enabled, obstacle_detection_enabled, left_wave_detected, performing_stunt
    
    # q键退出
    if key == ord('q'):
        print("用户请求降落")
        return True
        
    # r键重置目标
    elif key == ord('r'):
        target_id = None
        target_position = None
        position_history.clear()
        velocity_history.clear()
    
    # p键切换预测功能
    elif key == ord('p'):
        prediction_enabled = not prediction_enabled
    
    # o键切换避障功能
    elif key == ord('o'):
        obstacle_detection_enabled = not obstacle_detection_enabled
    
    # t键触发特技动作
    elif key == ord('t') and not performing_stunt:
        left_wave_detected = True
        performing_stunt = True
        threading.Thread(target=perform_stunt_sequence).start()
    
    # ESC键紧急降落
    elif key == 27:
        print("紧急降落!")
        return True
    
    return False

# 初始化无人机函数
def initialize_tello():
    """初始化和设置Tello无人机"""
    try:
        # 初始化Tello连接
        drone = Tello()
        drone.connect()
        
        # 防止自动降落的关键设置
        drone.RESPONSE_TIMEOUT = 15  # 增加响应超时时间
        drone.send_command_without_return("command")  # 进入SDK指令模式
        drone.set_speed(30)
        
        # 显示电池和版本信息
        battery = drone.get_battery()
        print(f"电池电量: {battery}%")
        
        # 启动视频流
        drone.streamon()
        
        # 起飞
        drone.takeoff()
        time.sleep(1)
        
        return drone
    except Exception as e:
        print(f"无人机初始化错误: {e}")
        return None

# 主程序
def main():
    global tello, processing_active, display_active, right_wave_detected,hands_processing_active
    
    try:
        # 初始化无人机
        tello = initialize_tello()
        if tello is None:
            print("无人机初始化失败，退出程序")
            return
        
        # 启动视频捕获线程
        capture_thread = threading.Thread(target=video_capture_thread)
        capture_thread.daemon = True
        capture_thread.start()
        
        # 启动姿态检测线程
        detection_thread = threading.Thread(target=pose_detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        
        # 启动手部检测线程
        hands_thread = threading.Thread(target=hands_detection_thread)
        hands_thread.daemon = True
        hands_thread.start()
        
        # 启动显示线程
        disp_thread = threading.Thread(target=display_thread)
        disp_thread.daemon = True
        disp_thread.start()
        
        # 主线程等待
        start_time = time.time()
        while True:
            # 如果检测到右手挥手，降落并退出
            if right_wave_detected:
                print("检测到右手挥手手势，准备降落...")
                tello.land()
                time.sleep(1)
                break
            
            # 检查命令间隔，确保定期发送命令
            current_time = time.time()
            if current_time - last_command_time > MAX_COMMAND_INTERVAL:
                send_control_command(0, 0, 0, 2 if keepalive_counter % 2 == 0 else -2)
            
            # 电池检查
            battery = tello.get_battery()
            if battery < 15:
                print(f"警告: 电池电量低 ({battery}%)，准备降落")
                break
            
            # 避免过高CPU使用率
            time.sleep(0.1)
            
            # 检查线程是否还活着
            if not capture_thread.is_alive() or not detection_thread.is_alive() or not disp_thread.is_alive():
                print("一个或多个线程已终止，准备降落")
                break
    
    finally:
        # 设置标志通知线程停止
        processing_active = False
        display_active = False
        hands_processing_active = False
        # 确保无人机降落
        print("程序结束，正在降落...")
        if tello:
            tello.land()
        
        # 等待线程结束
        time.sleep(2)
        
        # 释放资源
        cv2.destroyAllWindows()
        if tello:
            tello.streamoff()
        print("程序已安全退出")

if __name__ == "__main__":
    # 初始化 MediaPipe
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # 运行主程序
    main()
