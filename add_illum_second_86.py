import numpy as np
import cv2
import os

coord_path = './coordinates/'
img_path = '/home/test/PycharmProjects/2020/data/Shi_GeherDataSet/white_image/86/'
argument_img_path = '/home/test/PycharmProjects/2020/data/Shi_GeherDataSet/argument_image/86/'

file_list = sorted(os.listdir(img_path))
print(len(file_list))
print(file_list)
label = np.load('./Generate_label_86.npy')
print(len(label))


for i in range(label.shape[0]):
    image = np.load(img_path + file_list[i % 86]).astype(np.float32)

    Gain_R = float(label[i][0]) / float(np.max(label[i]))
    Gain_G = float(label[i][1]) / float(np.max(label[i]))
    Gain_B = float(label[i][2]) / float(np.max(label[i]))

    image[:, :, 0] = Gain_R * image[:, :, 0]
    image[:, :, 1] = Gain_G * image[:, :, 1]
    image[:, :, 2] = Gain_B * image[:, :, 2]

    image = image + 1

    image = (np.clip(image / image.max(), 0, 1) * 65535.0).astype(np.uint16)

    if (i + 569) < 1000:
        save_path = argument_img_path + 'img' + "0%d.png" % (i + 569)
    else:
        save_path = argument_img_path + 'img' + "%d.png" % (i + 569)

    cv2.imwrite(save_path, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
#
#
