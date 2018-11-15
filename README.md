# self_drive
基于树莓派的人工智能自动驾驶小车    
Artificial intelligence automatic driving car based on raspberry pie    
github传送门：https://github.com/Timthony/self_drive
# 整体流程（Technological process）    
电机控制（Motor control）    
摄像头调试（Camera debugging）    
道路数据采集（Road data acquisition）    
搭建深度学习模型，参数调试（Build deep learning model, parameter debug）    
自动驾驶真实道路模拟（Real road simulation of automatic driving）    
参数最终调试（Final debugging of parameters）    

使用方法（Usage method）：    
1. 先将树莓派小车硬件组装好（assemble the raspberry cart hardware.）    
2. 使用zth_car_control.py来控制小车的前后左右移动，配合zth_collect_data.py来人工操作，使小车在自己制作的跑道进行数据采集。（该过程在树莓派进行）
（Use zth_car_control.py to control the front and rear movement of the car, and cooperate with zth_collect_data.py to operate manually, so that the car can collect data on its own runway. (the process is carried out in raspberry pie).）    
3. 数据采集完成以后使用zth_process_img.py来对采集的数据进行处理，之前当前先完成一些数据清洗的工作。（电脑上执行）
（After data acquisition is completed, zth_process_img.py is used to process the collected data, and some data cleaning work is completed before. (computer execution)）    
4. 使用神经网络模型对数据进行训练zth_train.py，得到训练好的模型。（电脑上执行）
（ using neural network model to train data, zth_train.py, get a trained model. (computer execution)）    
5. 在树莓派小车上使用zth_drive和训练好的模型，载入模型，即可实现在原先跑道的自动驾驶。（树莓派上执行）    
（Auto-driving on the original runway can be realized by using zth_drive and trained model in the raspberry dispatch car and loading the model. (raspberry pie execution)）    
注意：只需要使用上述提到的代码即可，别的都是一些初始版本或者正在增加的一些新模块。 
（Note: All you need to do is use the code mentioned above. Others are original versions or new modules that are being added.）    
![](https://ws3.sinaimg.cn/large/006tNbRwly1fw2xwxarf2j30u0140tbg.jpg)    
![](https://ws4.sinaimg.cn/large/006tNbRwly1fx7tmmq5jjj30u00hgq6z.jpg)    

# 注意事项（Matters needing attention）：    
1. 赛道需要自己制作，很重要，决定了数据质量。(我是在地板上，贴的有色胶带，然后贴成了跑道的形状)。
（the track needs to be produced by itself, which is very important and determines the quality of data. (I was on the floor, taped with colored tape, and then pasted into the shape of the runway).）    
2. 赛道的宽度大约是车身的两倍。
（the width of the track is about two times that of the body.）    
3. 大约采集了五六万张图像，然后筛选出三四万张。
（about fifty thousand or sixty thousand images were collected and thirty thousand or forty thousand were screened out.）    
4. 摄像头角度问题
（camera angle problem）    
# 具体制作流程（Specific production process）： 
1. 小车原始模型，某宝购买玩具车即可，比如：有电机，有自带电池盒（给电机供电）
2. 树莓派，摄像头，蓄电电池组（用于树莓派供电）
3. 使用一些螺栓，螺柱，亚克力板将树莓派，蓄电电池固定在小车上（具体方法，看手头的工具吧）
4. 组装好以后，树莓派通过VNC连接电脑，登陆树莓派，在树莓派安装keras环境，以便最后调用训练好的模型。
5. 关于小车的控制（电机控制，摄像头采集数据），都在源文件，有注释，大致思路就是通过方向键AWSD来控制方向，使用了pygame的工具包。
6. 通过电脑端的wasd方向键手动控制小车（已经VNC连接好）在制作好的赛道上进行图像采集，直线部分按w，左拐弯按a，右拐弯按d等，建议采集50000张以上。
（采集的图像命名要求为，0_xxxx,1_xxxx,其中首位字母就代表了你按下的是哪个键，比如图像是0开头，那么这张图像就是直行，按下的是w键，这些0，1，2，3，4 数字就相当于数据的标签值）
7. 将图片从树莓派拷贝下来，进行数据清洗，使用电脑端的深度学习环境进行模型训练，使用的模型可以自行定义。
8. 将训练好的模型文件.h5拷贝到树莓派，然后通过树莓派调用载入模型，即可处理实时的图像，并且根据图像预测出是0，1，2，3，4等数字，也就表示了树莓派该怎么移动,通过树莓派控制电机即可。


# 正在进行一些改进：    
1.使用迁移学习进行fine-tuning是否可以提高精度    
2.处理光照问题    
3.处理数据类别不平衡的问题    
欢迎交流讨论    


