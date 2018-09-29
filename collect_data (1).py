import io
import car_control
import os
os.environ['SDL_VIDEODRIVE'] = 'x11'
import pygame
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

    def write(self, buf):
        global key
        if buf.startswith(b'\xff\xd8'):
            # Start of new frame; close the old one (if any) and
            # open a new output
            if self.output:
                self.output.close()
            self.frame_num += 1
            self.output = io.open('%s_image%s.jpg' % (key,time()), 'wb')
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
    pygame.display.set_mode((1,1))
    car_control.car_stop()
    sleep(0.1)
    print("Start control!")
 
    while is_capture_running:
        # get input from human driver
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                key_input = pygame.key.get_pressed()
                print(key_input[pygame.K_w], key_input[pygame.K_a], key_input[pygame.K_d])
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    car_control.car_move_forward()
                elif key_input[pygame.K_a]:
                    print("Left")
                    car_control.car_turn_left()
                    sleep(0.1)
                    key = 0
                elif key_input[pygame.K_d]:
                    print("Right")
                    car_control.car_turn_right()
                    sleep(0.1)
                    key = 1
                elif key_input[pygame.K_s]:
                    print("Backward")
                    car_control.car_move_backward()
                    key = 3
                elif key_input[pygame.K_k]:
                    car_control.car_stop()
            elif event.type == pygame.KEYUP:
                key_input = pygame.key.get_pressed()
                if key_input[pygame.K_w] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Forward")
                    key = 2
                    car_control.car_turn_straight()
                    car_control.car_move_forward()
                elif key_input[pygame.K_s] and not key_input[pygame.K_a] and not key_input[pygame.K_d]:
                    print("Backward")
                    key = 3
                    car_control.car_move_backward()
                else:
                    print("Stop")
                    car_control.car_stop()
                #car_control.cleanGPIO()
    car_control.cleanGPIO()

if __name__ == '__main__':
    global train_labels, train_img, key

    print("capture thread")
    print ('-' * 50)
    capture_thread = threading.Thread(target=pi_capture,args=())
    capture_thread.setDaemon(True)
    capture_thread.start()
    
    my_car_control()

    while is_capture_running:
        pass

    print("Done!")
    car_control.car_stop()
    car_control.clean_GPIO()
