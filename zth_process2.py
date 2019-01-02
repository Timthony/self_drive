# coding=utf-8
# 将原始的jpg图片处理成Inception-v3模型需要的299×299×3的数字矩阵
# 将所有的图片分为训练/验证/测试3个数据集
import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import os

#####################################1.定义需要使用到的常量###########################

# 原始输入数据的目录
INPUT_DATA = 'datasets/training_data'
# 输出文件的地址。我们将整理后的图片数据通过numpy格式保存
OUTPUT_FILE = 'datasets/processed_data.npy'
# 测试数据和验证数据的比例
VALIDATION_PRECENTAGE = 10
TEST_PRECENTAGE = 10
#####################################2.定义数据处理过程###############################

# 读取数据并将数据分割成训练数据/验证数据/测试数据
def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True

    # 初始化各个数据集。
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有的子目录。
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取一个子目录中所有的图片文件。
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print("processing:", dir_name)

        i = 0
        # 处理图片数据。
        for file_name in file_list:
            i += 1
            # 读取并解析图片，将图片转化为299*299以方便inception-v3模型来处理。
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)

            # 随机划分数据聚。
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
            if i % 200 == 0:
                print(i, "images processed.")
        current_label += 1

    # 将训练数据随机打乱以获得更好的训练效果。
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])

def main():
    #config = tf.ConfigProto(allow_soft_placement = True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)) as sess:
        processed_data = create_image_lists(sess, TEST_PRECENTAGE, VALIDATION_PRECENTAGE)
        # 通过numpy格式储存处理过的数据
        np.save(OUTPUT_FILE, processed_data)
if __name__ == '__main__':
    main()





































