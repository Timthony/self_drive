# 搭建深度学习模型
# 导入库
# 自动驾驶模型真实道路模拟行驶
import keras
import tensorflow
import sys
import os
import h5py
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.models import load_model, Model, Input
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam, SGD

np.random.seed(0)

# 全局变量
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 120, 160, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)



# step1,载入数据，并且分割为训练和验证集
# 问题，数据集太大了，已经超过计算机内存
def load_data():
    # load
    image_array = np.zeros((1, 120, 160, 3))               # 初始化
    label_array = np.zeros((1, 5), 'float')
    training_data = glob.glob('training_data_npz/*.npz')
    # 匹配所有的符合条件的文件，并将其以list的形式返回。
    print("匹配完成。开始读入")
    print("一共%d轮", len(training_data))

    # if no data, exit，容错判断
    if not training_data:
        print("No training data in directory, exit")
        sys.exit()
    i = 0
    for single_npz in training_data:
        with np.load(single_npz) as data:
            print(data.keys())
            i = i + 1
            print("在打印关键值", i)
            train_temp = data['train_imgs']
            train_labels_temp = data['train_labels']
        image_array = np.vstack((image_array, train_temp)) # 把文件读取都放入，内存
        label_array = np.vstack((label_array, train_labels_temp))
        print("第%d轮完成", i)
    print("循环完了")
    X = image_array[1:, :]
    y = label_array[1:, :]
    print('Image array shape: ' + str(X.shape))
    print('Label array shape: ' + str(y.shape))
    print(np.mean(X))
    print(np.var(X))

    # now we can split the data into a training (80), testing(20), and validation set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    return X_train, X_valid, y_train, y_valid


# step2 建立模型
def build_model(keep_prob):
    print("开始编译模型")
    model = Sequential()
    model.add(Lambda(lambda x: (x/102.83 - 1), input_shape = INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
    model.add(Conv2D(64, (3, 3),activation='elu'))
    model.add(Conv2D(64, (3, 3),activation='elu'))
    model.add(Dropout(keep_prob))  # Dropout将在训练过程中每次更新参数时随机断开一定百分比（p）的输入神经元连接
    model.add(Flatten())
    #model.add(Dense(500, activation='elu'))
    model.add(Dense(250, activation='elu'))
    #model.add(Dense(50, activation='elu'))
    model.add(Dense(5))
    model.summary()

    return model

# step3 训练模型
def train_model(model, learning_rate, nb_epoch, samples_per_epoch,
                batch_size, X_train, X_valid, y_train, y_valid):
    # 值保存最好的模型存下来
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min')
    # EarlyStopping patience：当earlystop被激活（如发现loss相比上一个epoch训练没有下降），
    # 则经过patience个epoch后停止训练。
    # mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
    early_stop = EarlyStopping(monitor='loss', min_delta=.0005, patience=10,
                               verbose=1, mode='min')
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=20, write_graph=True,write_grads=True,
                              write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                              embeddings_metadata=None)
    # 编译神经网络模型，loss损失函数，optimizer优化器， metrics列表，包含评估模型在训练和测试时网络性能的指标
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    # 训练神经网络模型，batch_size梯度下降时每个batch包含的样本数，epochs训练多少轮结束，
    # verbose是否显示日志信息，validation_data用来验证的数据集
    model.fit_generator(batch_generator(X_train, y_train, batch_size),
                        steps_per_epoch=samples_per_epoch/batch_size,
                        epochs = nb_epoch,
                        max_queue_size=1,
                        validation_data=batch_generator(X_valid, y_valid, batch_size),
                        validation_steps=len(X_valid)/batch_size,
                        callbacks=[tensorboard, checkpoint, early_stop],
                        verbose=2)

# step4
# 可以一个batch一个batch进行训练，CPU和GPU同时开工
def batch_generator(X, y, batch_size):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty([batch_size, 5])
    while True:
        i = 0
        for index in np.random.permutation(X.shape[0]):
            images[i] = X[index]
            steers[i] = y[index]
            i += 1
            if i == batch_size:
                break
        yield (images, steers)


# step5 评估模型
#def evaluate(x_test, y_test):
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])


def main():
    # 打印出超参数

    print('-'*30)
    print('parameters')
    print('-'*30)


    keep_prob = 0.5
    learning_rate = 0.0001
    nb_epoch = 100
    samples_per_epoch = 3000
    batch_size = 30

    print('keep_prob = ', keep_prob)
    print('learning_rate = ', learning_rate)
    print('nb_epoch = ', nb_epoch)
    print('samples_per_epoch = ', samples_per_epoch)
    print('batch_size = ', batch_size)
    print('-' * 30)

    # 开始载入数据
    data = load_data()
    print("数据加载完毕")
    # 编译模型
    model = build_model(keep_prob)
    # 在数据集上训练模型，保存成model.h5
    train_model(model, learning_rate, nb_epoch, samples_per_epoch, batch_size, *data)
    print("模型训练完毕")


if __name__ == '__main__':
    main()






