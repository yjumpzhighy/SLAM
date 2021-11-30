
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import matplotlib.pyplot as plt



class KF(object):
  def __init__(self):
    #[x,y,z,vx,vy,vz]
    self.dim_states = 6
    #[x,y,z]
    self.dim_measures = 3
    self.kf = KalmanFilter(dim_x=self.dim_states, dim_z=self.dim_measures)
    self.fps = 15
    self.dt = 1. / np.float32(self.fps)
    #initialization 
    self.kf.x = np.zeros((self.dim_states,1))
    #constant velocity motion model
    self.kf.F = np.array([[1,0,0,self.dt,0,0],
                          [0,1,0,0,self.dt,0],
                          [0,0,1,0,0,self.dt],
                          [0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1]], dtype=np.float32)
    #[dim_measures, dim_states]
    self.kf.H = np.array([[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0]], dtype=np.float32)
    self.kf.P = np.array([[5., 0,  0,  0,   0,  0],
                          [0,  3., 0,  0,   0,  0],
                          [0,  0,  3., 0,   0,  0],
                          [0,  0,  0,  5., 0,  0],
                          [0,  0,  0,  0,   5., 0],
                          [0,  0,  0,  0,   0,  5.]], dtype=np.float32)
    self.kf.Q = np.array([[2., 0,   0,   0,   0,  0],
                          [0,  0.1, 0,   0,   0,  0],
                          [0,  0,   0.1, 0,   0,  0],
                          [0,  0,   0,   10., 0,  0],
                          [0,  0,   0,   0,   5., 0],
                          [0,  0,   0,   0,   0,  5.]], dtype=np.float32)
    self.kf.R = np.array([[5., 0,  0],
                          [0,  2., 0],
                          [0,  0,  2.]], dtype=np.float32)
                        
  def Predict(self):
    self.kf.predict()

  def Update(self, z):
    self.kf.update(z)




test_iter = 1000
def Test():
    x = np.array([0,0,0,2.0,0.1,0.1],dtype=np.float32).reshape(6,1)
    dt=0.1
    F = np.array([[1,0,0,dt,0,0],
                  [0,1,0,0,dt,0],
                  [0,0,1,0,0,dt],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]], dtype=np.float32)
    Q = 0.01 * np.array([[0.01, 0,   0,   0,   0,  0],
                        [0,  0.01, 0,   0,   0,  0],
                        [0,  0,   0.01, 0,   0,  0],
                        [0,  0,   0,   0.005, 0,  0],
                        [0,  0,   0,   0,   0.005, 0],
                        [0,  0,   0,   0,   0,  0.005]], dtype=np.float32)
    R = np.array([0.2, 0.1,  0.1], dtype=np.float32)
    
    real_state = []
    for i in range(test_iter):
      real_state.append(x[0:3].T)
      x = np.dot(F, x) + np.random.multivariate_normal(
                          mean=(0,0,0,0,0,0),cov=Q).reshape(6,1)

    measurements = [x+np.random.normal(0,R) for x in real_state]
    
    return measurements, real_state

def plot_result(measurements,filter_result, real_state):
    plt.figure(figsize=(8,4))
    
    measurements_x = np.array([m[0][0] for m in measurements])
    measurements_y = np.array([m[0][1] for m in measurements])
    measurements_z = np.array([m[0][2] for m in measurements])
    plt.plot(measurements_x, measurements_y, label = 'Measurements')

    filter_result_x = np.array([m[0] for m in filter_result])
    filter_result_y = np.array([m[1] for m in filter_result])
    filter_result_z = np.array([m[2] for m in filter_result])
    plt.plot(filter_result_x, filter_result_y, label = 'Kalman Filter')



    real_state_x = np.array([m[0][0] for m in real_state])
    real_state_y = np.array([m[0][1] for m in real_state])
    real_state_z = np.array([m[0][2] for m in real_state])
    plt.plot(real_state_x, real_state_y, label = 'real states')

    plt.legend()
    plt.show()

    # plt.plot(range(1,len(real_state)), real_state[1:], label = 'Real statement' )
    # plt.plot(range(1,len(filter_result)), np.array(filter_result)[1:,0], label = 'Kalman Filter')
    # plt.legend()
    # plt.xlabel('Time',fontsize=14)
    # plt.ylabel('velocity [m]',fontsize=14)
    # plt.show()
    
    # plt.figure(figsize=(8,4))
    # plt.axhline(5, label='Real statement') #, label='$GT_x(real)$'
    # plt.plot(range(1,len(filter_result)), np.array(filter_result)[1:,1], label = 'Kalman Filter')
    # plt.legend()
    # plt.xlabel('Time',fontsize=14)
    # plt.ylabel('velocity [m]',fontsize=14)
    plt.show()



if __name__ == "__main__":
  mykf = KF()
  measurements, real_state = Test()
    
  results=list()
  results.append(mykf.kf.x)
  for i in range(1,test_iter):
        z = measurements[i]
        mykf.kf.predict()
        mykf.kf.update(z)
        results.append(mykf.kf.x)

  plot_result(measurements, results, real_state)
