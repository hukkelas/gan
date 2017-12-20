import math
import matplotlib.pyplot as plt
import numpy as np
def save_image(fname, images, row_dim=3):
  h,w,c_dim = images.shape[1:]
  col_dim = math.ceil(images.shape[0] / row_dim)
  result = np.ones((images.shape[1]*row_dim,col_dim*images.shape[2], images.shape[3]))
  row_idx=0
  col_idx=0
  for i in range(len(images)):
    result[row_idx*h:(row_idx +1)*h, col_idx*w:(col_idx+1)*w] = images[i]
    row_idx += 1

    if row_idx == row_dim:
      row_idx = 0
      col_idx += 1
  if result.shape[-1] == 1:
    plt.imsave(fname, result[:,:,0], cmap="gray")
  else:
    plt.imsave(fname, result)
