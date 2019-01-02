# 收集数据，赛道照片和对应的前、后、左、右、4停
# 对应图片和相应的标签值
import io
import zth_car_control
import os
os.environ['SDL_VIDEODRIVE'] = 'x11'
import pygame     # 检测模块
from time import ctime,sleep,time
import threading
import numpy as np
import picamera
import picamera.array

global train_labels, train_img, is_capture_running, key

class SplitFrames(object):
    
    def __init__(self):
        self.frame_num = 0
        self.output = None
# 处理图像的函数write
# 对视频拍摄的每一帧进行处理，构造一个自定义输出类，每拍摄一帧都会进来write处理
    def write(self, buf):
        global key
        if buf.startswith(b'\xff\xd8'):                            # 代表一个JPG图片的开始，新照片的开头
            # Start of new frame; close the old one (if any) and
            # open a new output
            if self.output:
                self.output.close()
            self.frame_num += 1
            self.output = io.open('%s_image%s.jpg' % (key,time()), 'wb')           # 改变格式为jpg
        self.output.write(buf)
    
 
def pi_capture():
    global train_img, is_capture_running,train_labels,key
    
    #init the train_label array
    print("Start capture")        
    is_capture_running = True

    with picamera.PiCamera(resolution=(160, 120), framerate=30) as camera:
        # 根据摄像头实际情况判断是否要加这句上下翻转
        # camera.vflip = True
        # Give the camera some warm-up time
        sleep(2)
        output = SplitFrames()
        start = time()
        camera.start_recording(output, format='mjpeg')
        camera.wait_recording(120)
        camera.stop_recording()
        finish = time()
    print('Captured %d frames at %.2ffps' % (
        output.frame_num,
        output.frame_num / (finish - start)))
    
    print("quit pi capture")
    is_capture_running = False

def my_car_control(): 
    global is_capture_running, key
    key = 4
    pygame.init()
    pygame.display.set_mode((1,1))            # 窗口
    zth_car_control.car_stop()
    sleep(0.1)
    print("Start control!")
 
    while is_capture_running:
        # get input from human driver
        # 
        for event in pygame.event.get():
            # 判断事件是不是按键按下的事件
            if event.type == pygame.KEYDOWN:  
                key_input = pygame.key.get_pressed()     # 可以同时检测多个按键
                print(key_input[pygame.K_w], key_input[pygame.K_a], key_input[pygame.K_d])
                # 按下前进，保存图片以2开头
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2 
                    zth_car_control.car_move_forward()
                # 按下左键，保存图片以0开头
                elif key_input[pygame.K_a]:
                    print("Left")
                    zth_car_control.car_turn_left()
                    sleep(0.1)
                    key = 0
                # 按下d右键，保存图片以1开头
                elif key_input[pygame.K_d]:
                    print("Right")
                    zth_car_control.car_turn_right()
                    sleep(0.1)
                    key = 1
                # 按下s后退键，保存图片为3开头
                elif key_input[pygame.K_s]:
                    print("Backward")
                    zth_car_control.car_move_backward()
                    key = 3
                # 按下k停止键，停止
                elif key_input[pygame.K_k]:
                    zth_car_control.car_stop()
            # 检测按键是不是抬起
            elif event.type == pygame.KEYUP:
                key_input = pygame.key.get_pressed()
                # w键抬起，轮子回正
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    zth_car_control.car_turn_straight()
                    zth_car_control.car_move_forward()
                # s键抬起
                elif key_input[pygame.K_s] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Backward")
                    key = 3
                    zth_car_control.car_move_backward()
                else:
                    print("Stop")
                    zth_car_control.car_stop()
                #car_control.cleanGPIO()
    zth_car_control.clean_GPIO()

if __name__ == '__main__':
    global train_labels, train_img, key

    print("capture thread")
    print('-' * 50)
    capture_thread = threading.Thread(target=pi_capture,args=())   # 开启线程
    capture_thread.setDaemon(True)
    capture_thread.start()
    
    my_car_control()

    while is_capture_running:
        pass

    print("Done!")
    zth_car_control.car_stop()
    zth_car_control.clean_GPIO()
