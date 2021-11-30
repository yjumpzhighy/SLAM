import numpy as np


class CameraConfig(object):
  def __init__(self, dataset):
    if dataset == 'default':
      self.W = 960
      self.H = 540
      self.F = 270  #focal length in pixel
      self.K = np.array([[self.F, 0, self.W//2],
                         [0, self.F, self.H//2],
                         [0, 0, 1]])
    elif dataset == 'kitti_tracking':
      self.W = 1242
      self.H = 375
      filepath = '/home/zuyuan/Data/kitti_tracking/calib/training/calib/0000.txt'
      calib = self.read_kitti_calib(filepath)
      self.K = calib[:3,:3]

  def read_kitti_calib(self, calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
      if i == 2:
        calib = np.array(line.strip().split(' ')[1:], dtype=np.float32)
        calib = calib.reshape(3, 4)
        return calib
