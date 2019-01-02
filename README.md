# self_drive
基于树莓派的人工智能自动驾驶小车    

使用方法：    
1. 先将树莓派小车硬件组装好
2. 使用zth_car_control.py来控制小车的前后左右移动，配合zth_collect_data.py来人工操作，使小车在自己制作的跑道进行数据采集。（该过程在树莓派进行）
3. 数据采集完成以后使用zth_process_img.py来对采集的数据进行处理，之前当前先完成一些数据清洗的工作。（电脑上执行）
4. 使用神经网络模型对数据进行训练zth_train.py，得到训练好的模型。（电脑上执行）
5. 在树莓派小车上使用zth_drive和训练好的模型，载入模型，即可实现在原先跑道的自动驾驶。（树莓派上执行）

![](https://ws3.sinaimg.cn/large/006tNbRwly1fw2xwxarf2j30u0140tbg.jpg)    

