import numpy as np

RGB = 0.14 + (0.85 - 0.14) * np.random.rand(430, 3)
RGB_L2_NORM = np.linalg.norm(RGB, axis=1)[..., np.newaxis]
print(len(RGB_L2_NORM))
RGB = RGB / RGB_L2_NORM
print(RGB)
#
np.save('Generate_label_86.npy', RGB)