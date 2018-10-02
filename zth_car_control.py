import RPi.GPIO as GPIO
import time

# 设定速度，满电时速度太快，图像处理速度跟不上
# 直行快一点，转向慢一点
speed1 = 60          # 直行速度
speed2 = 50          # 拐弯速度
# 轮子定义
backMotorinput1 = 7   #后轮1
backMotorinput2 = 11   #后轮2

frontMotorinput1 = 15    #前轮1
frontMotorinput2 = 13    #前轮2

backMotorEn = 12    #使能端口1
frontMotorEn = 16    #使能端口2

GPIO.setmode(GPIO.BOARD)                         # 设置模式
GPIO.setup(backMotorinput1,GPIO.OUT)             # 此端口为输出模式
GPIO.setup(backMotorinput2,GPIO.OUT)             # 此端口为输出模式
GPIO.setup(frontMotorinput1,GPIO.OUT)            # 此端口为输出模式
GPIO.setup(frontMotorinput2,GPIO.OUT)            # 此端口为输出模式
GPIO.setup(backMotorEn,GPIO.OUT)
GPIO.setup(frontMotorEn,GPIO.OUT)
# 将控制小车运动封装为函数
backMotorPwm = GPIO.PWM(backMotorEn,100)         # 配置PWM
backMotorPwm.start(0)                            # 开始输出PWM
# 当使能端口输入低电压时，电机驱动板将不对电机输出电流，电机将不工作。
# 当使能端口输入高电压时，让前轮转向电机正常工作。
# 向前走
def car_move_forward():
	GPIO.output(backMotorinput1,GPIO.HIGH)
	GPIO.output(backMotorinput2,GPIO.LOW)
	backMotorPwm.ChangeDutyCycle(speed1)         # 改变PWM占空比，参数为占空比
# 向后退
def car_move_backward():
	GPIO.output(backMotorinput1,GPIO.LOW)
	GPIO.output(backMotorinput2,GPIO.HIGH)
	backMotorPwm.ChangeDutyCycle(speed2)
# 左拐
def car_turn_left():
	GPIO.output(frontMotorEn,GPIO.HIGH)    # 当使能端口输入高电压时，让前轮转向电机正常工作。
	GPIO.output(frontMotorinput1,GPIO.HIGH)
	GPIO.output(frontMotorinput2,GPIO.LOW)
	GPIO.output(backMotorinput1,GPIO.HIGH)
	GPIO.output(backMotorinput2,GPIO.LOW)
	backMotorPwm.ChangeDutyCycle(speed2)
# 右拐	
def car_turn_right():
	GPIO.output(frontMotorEn,GPIO.HIGH)    # 当使能端口输入高电压时，让前轮转向电机正常工作。
	GPIO.output(frontMotorinput1,GPIO.LOW)
	GPIO.output(frontMotorinput2,GPIO.HIGH)
	GPIO.output(backMotorinput1,GPIO.HIGH)
	GPIO.output(backMotorinput2,GPIO.LOW)
	backMotorPwm.ChangeDutyCycle(speed2)
# 
def carbackleft():
	GPIO.output(frontMotorEn,GPIO.HIGH)    # 当使能端口输入高电压时，让前轮转向电机正常工作。
	GPIO.output(frontMotorinput1,GPIO.HIGH)
	GPIO.output(frontMotorinput2,GPIO.LOW)
	GPIO.output(backMotorinput1,GPIO.LOW)
	GPIO.output(backMotorinput2,GPIO.HIGH)
	backMotorPwm.ChangeDutyCycle(speed2)
# 
def carbackRight():
	GPIO.output(frontMotorEn,GPIO.HIGH)    # 当使能端口输入高电压时，让前轮转向电机正常工作。
	GPIO.output(frontMotorinput1,GPIO.LOW)
	GPIO.output(frontMotorinput2,GPIO.HIGH)
	GPIO.output(backMotorinput1,GPIO.LOW)
	GPIO.output(backMotorinput2,GPIO.HIGH)
	backMotorPwm.ChangeDutyCycle(speed2)
# 清除
def clean_GPIO():
	GPIO.cleanup()
	backMotorPwm.stop()                          # 停止输出PWM
# 前轮回正函数
def car_turn_straight():
	GPIO.output(frontMotorEn,GPIO.LOW)    # 当使能端口输入低电压时，电机驱动板将不对电机输出电流，电机将不工作。
	time.sleep(0.05)
# 停止
def car_stop():
	GPIO.output(backMotorinput1,GPIO.LOW)
	GPIO.output(backMotorinput2,GPIO.LOW)
	GPIO.output(frontMotorEn,GPIO.LOW)

if __name__ == '__main__':
	car_turn_straight()
	car_move_forward()
	time.sleep(10)
	clean_GPIO()
