import scipy
from scipy import io
import numpy as np
import cv2

# label_file = scipy.io.loadmat('./train.mat')
# label = label_file['real_rgb']
# print(label.shape)
# label_norm = np.linalg.norm(label, axis=1)
# temp = np.zeros(shape=[554, 3])
# for i in range(3):
#     temp[:,i] = label_norm
#
# print(temp)
# label_train = label / temp
# print(label_norm.shape)
# print(label)
# print(label_train)
#
# np.save('label.npy', label_train)

label_Argument = np.load('./Generate_label_86.npy')
print(label_Argument)

r = label_Argument[:,0]
g = label_Argument[:,1]
b = label_Argument[:,2]

image = np.zeros(shape=[430, 430, 3])
#
for i in range(430):
    image[:, i, 0] = r
    image[:, i, 1] = g
    image[:, i, 2] = b

cv2.imwrite('Generate_label.png', image * 255.)