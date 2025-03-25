import time
import threading
import cv2
import numpy as np
import mediapipe as mp
from djitellopy import Tello

# 初始化MediaPipe解决方案
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 初始化Tello无人机
tello = Tello()
tello.connect()
print(f"电量: {tello.get_battery()}%")

# 创建人体姿势和手势检测实例
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=2)

# 全局变量
frame = None
frame_lock = threading.Lock()
take_photo = False
photo_taken = False
photo_timer = None
head_detected = False
position_locked = False

def reset_photo_taken():
    """重置拍照标志"""
    global photo_taken
    photo_taken = False

def video_stream():
    """视频流线程，负责显示摄像头画面和处理检测"""
    global frame, take_photo, photo_taken, photo_timer, head_detected, position_locked
    
    # 启动视频流
    tello.streamon()
    
    # 等待视频流稳定
    time.sleep(2)
    
    # 初始状态：保持当前位置
    tello.send_rc_control(0, 0, 0, 0)
    
    while True:
        try:
            # 获取Tello帧
            tello_frame = tello.get_frame_read().frame
            
            if tello_frame is None:
                continue
            
            # 创建用于显示的帧副本
            display_frame = tello_frame.copy()
            
            # 使用MediaPipe处理帧
            with frame_lock:
                # 将BGR转换为RGB用于MediaPipe
                rgb_frame = cv2.cvtColor(tello_frame, cv2.COLOR_BGR2RGB)
                
                # 进行姿势检测
                pose_results = pose.process(rgb_frame)
                
                # 重置头部检测状态
                current_head_detected = False
                
                # 检查是否检测到人体骨架
                if pose_results.pose_landmarks:
                    # 在显示帧上绘制骨架
                    mp_drawing.draw_landmarks(
                        display_frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                    
                    # 检查头部关键点是否存在
                    # 使用鼻子点（点0）作为头部标志
                    nose = pose_results.pose_landmarks.landmark[0]
                    if nose.visibility > 0.5:  # 如果鼻子可见性高于阈值
                        current_head_detected = True
                        
                        # 在显示帧上标记鼻子位置
                        h, w, c = display_frame.shape
                        nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                        cv2.circle(display_frame, (nose_x, nose_y), 10, (0, 255, 0), -1)
                
                # 更新头部检测状态
                head_detected = current_head_detected
                
                # 显示头部检测状态
                if head_detected:
                    cv2.putText(display_frame, "头部已检测", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 一旦检测到头部，锁定位置
                    if not position_locked:
                        position_locked = True
                        tello.send_rc_control(0, 0, 0, 0)  # 停止移动
                        print("位置已锁定")
                    
                    cv2.putText(display_frame, "位置已锁定", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "未检测到头部", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 只有锁定位置后才检测手势
                peace_sign_detected = False
                if position_locked:
                    # 进行手势检测
                    hand_results = hands.process(rgb_frame)
                    
                    if hand_results.multi_hand_landmarks:
                        for hand_landmarks in hand_results.multi_hand_landmarks:
                            # 在显示帧上绘制手部标记
                            mp_drawing.draw_landmarks(
                                display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                            # 获取标记位置
                            landmarks = hand_landmarks.landmark
                            
                            # 检测剪刀手（比耶）
                            # 检查食指和中指是否伸出，其他手指弯曲
                            if (landmarks[8].y < landmarks[5].y and  # 食指伸出
                                landmarks[12].y < landmarks[9].y and  # 中指伸出
                                landmarks[16].y > landmarks[13].y and  # 无名指弯曲
                                landmarks[20].y > landmarks[17].y):    # 小指弯曲
                                
                                peace_sign_detected = True
                                cv2.putText(display_frame, "剪刀手检测到", (10, 90), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 检测到剪刀手时拍照
                if head_detected and peace_sign_detected and position_locked and not photo_taken:
                    take_photo = True
                    photo_taken = True
                    
                    # 5秒后重置拍照标志
                    if photo_timer:
                        photo_timer.cancel()
                    photo_timer = threading.Timer(5.0, reset_photo_taken)
                    photo_timer.daemon = True
                    photo_timer.start()
                
                # 保存不带标注的原始帧用于拍照
                frame = tello_frame.copy()
            
            # 显示带标注的帧
            cv2.imshow('Tello Camera Feed', display_frame)
            
            # 按'q'键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        except Exception as e:
            print(f"视频流错误: {e}")
            time.sleep(0.1)
    
    # 清理
    tello.streamoff()
    cv2.destroyAllWindows()

def drone_control():
    """无人机控制线程，负责拍照和执行翻滚"""
    global take_photo
    
    while True:
        try:
            if take_photo:
                print("准备拍照...")
                # 停止移动以稳定拍照
                tello.send_rc_control(0, 0, 0, 0)
                time.sleep(0.5)  # 稳定无人机
                
                # 拍照
                with frame_lock:
                    if frame is not None:
                        try:
                            # 转换为RGB（按要求）
                            rgb_photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # 使用时间戳保存照片
                            timestamp = int(time.time())
                            photo_path = f"tello_photo_{timestamp}.jpg"
                            cv2.imwrite(photo_path, rgb_photo)
                            print(f"照片已保存为 {photo_path}")
                            
                            # 照片拍摄成功后执行翻滚动作
                            try:
                                time.sleep(0.5)  # 短暂等待以确保照片已保存
                                print("执行翻滚...")
                                tello.flip_back()
                                print("无人机已翻滚，表示照片已拍")
                            except Exception as e:
                                print(f"翻滚失败: {e}")
                        except Exception as e:
                            print(f"保存照片错误: {e}")
                        
                        # 重置标志
                        take_photo = False
        except Exception as e:
            print(f"无人机控制错误: {e}")
            
        # 休眠以减少CPU使用
        time.sleep(0.1)

def main():
    """主函数"""
    try:
        # 初始化无人机
        tello.connect()
        print(f"电量: {tello.get_battery()}%")
        
        # 启动视频流线程（在起飞前启动，以便可以在控制台中看到视频）
        video_thread = threading.Thread(target=video_stream)
        video_thread.daemon = True
        video_thread.start()
        print("视频流线程已启动")
        
        # 等待视频流稳定
        time.sleep(3)
        
        # 起飞
        tello.takeoff()
        print("无人机已起飞")
        
        # 暂停一下，让无人机稳定
        tello.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        
        # 启动无人机控制线程
        control_thread = threading.Thread(target=drone_control)
        control_thread.daemon = True
        control_thread.start()
        print("无人机控制线程已启动")
        
        # 保持主线程活动
        print("系统运行中。按Ctrl+C停止。")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("程序被用户停止")
    except Exception as e:
        print(f"意外错误: {e}")
    finally:
        # 程序退出前降落无人机
        try:
            # 在降落前先悬停
            tello.send_rc_control(0, 0, 0, 0)
            time.sleep(1)
            
            tello.land()
            print("无人机安全降落")
        except:
            print("紧急情况：无人机降落失败")
            tello.emergency()  # 如果降落失败，紧急停止
        
        # 清理剩余线程
        if photo_timer:
            photo_timer.cancel()

if __name__ == "__main__":
    main()