import numpy as np
import os
import cv2
from scipy import io

coord_path = './coordinates/'
img_path = '/home/test/PycharmProjects/2020/data/Shi_GeherDataSet/image86/'
white_img_path = '/home/test/PycharmProjects/2020/data/Shi_GeherDataSet/white_image/86/'

file_list = sorted(os.listdir(img_path))
ground_truth = io.loadmat('./real_illum_568.mat')['real_rgb'][:86]
print(file_list)

def get_mcc_coord(img_name):
    file_path = coord_path + (img_name.split('/')[-1]).split('.')[0] + '_macbeth.txt'
    with open(file_path, 'r') as f:
        lines = f.readlines()
        width, height = map(float, lines[0].split())
        scale_x = 1 / width
        scale_y = 1 / height
        lines = [lines[1], lines[2], lines[4], lines[3]]
        polyen = []
        for line in lines:
            line = line.strip().split()
            x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
            polyen.append((x, y))
        return np.array(polyen, dtype='float32')

def load_image(img_name):
    img = cv2.imread(img_path + img_name, cv2.IMREAD_UNCHANGED)
    img = np.array(img, dtype='float32')

    if img_name.__contains__('IMG'):
        black_level = 129
        print('Error!!!')
    else:
        black_level = 1

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.maximum(img - black_level, [0, 0, 0])
    return img

def load_img_without_mcc(img_name):
    img = load_image(img_name)
    # img = (np.clip(img / img.max(), 0, 1) * 65535.0).astype(np.uint16)
    polygon = get_mcc_coord(img_name) * np.array([img.shape[1], img.shape[0]])
    polygon = polygon.astype(np.int32)
    cv2.fillPoly(img, [polygon], (1e-5,)*3)
    return img

for i in range(len(file_list)):
    img = load_img_without_mcc(file_list[i])

    Gain_R = float(np.max(ground_truth[i])) / float(ground_truth[i][0])
    Gain_G = float(np.max(ground_truth[i])) / float(ground_truth[i][1])
    Gain_B = float(np.max(ground_truth[i])) / float(ground_truth[i][2])

    img[:, :, 0] = Gain_R * img[:, :, 0]
    img[:, :, 1] = Gain_G * img[:, :, 1]
    img[:, :, 2] = Gain_B * img[:, :, 2]

    if (i + 1) < 10:
        save_path = white_img_path + 'img' + "000%d.png"%(i+1)
    elif(i + 1) < 100:
        save_path = white_img_path + 'img' + "00%d.png" % (i + 1)
    elif (i + 1) < 1000:
        save_path = white_img_path + 'img' + "0%d.png" % (i + 1)
    else:
        save_path = white_img_path + 'img' + "%d.png" % (i + 1)

    np.save(save_path, img)