import os
import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform
from multiprocessing import Process, Queue
import OpenGL.GL as gl
import pangolin

from cameraconfig import CameraConfig 
from orb.orb_extractor import OrbExtractor
from kf.kalman_filter import KF



class Map(object):
  #construct map to show the keypoints 3d point cloud and camera pose
  def __init__(self, W, H):
    self.width = W
    self.height = H
    self.poses = []
    self.points = []
    self.state = None
    self.q = Queue()
    p = Process(target=self.viewer_thread, args=(self.q,))
    p.daemon = True
    p.start()

  def viewer_thread(self, q):
    self.viewer_init()
    while True:
      self.viewer_refresh(q)

  def viewer_init(self):
    pangolin.CreateWindowAndBind('Main', self.width, self.height)
    gl.glEnable(gl.GL_DEPTH_TEST)
    self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.width, self.height, 420, 420, 
                                      self.width//2, self.height//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -8,
                                     0,   0,  0,
                                     0,  -1,  0))
    self.handler = pangolin.Handler3D(self.scam)
    # Create Interactive View in window
    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.width/self.height)
    self.dcam.SetHandler(self.handler)

  def viewer_refresh(self, q):
    if self.state is None or not q.empty():
      self.state = q.get()
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(1.0, 1.0, 1.0, 1.0)
    self.dcam.Activate(self.scam)
    # draw poses
    gl.glColor3f(0.0, 1.0, 0.0)
    pangolin.DrawCameras(self.state[0])
    # draw keypoints
    gl.glPointSize(2)
    gl.glColor3f(1.0, 0.0, 0.0)
    pangolin.DrawPoints(self.state[1])

    pangolin.FinishFrame()
  
  def add_observation(self, pose, points):
        self.poses.append(pose)
        for point in points:
            self.points.append(point)

  def display(self):
    poses = np.array(self.poses)
    points = np.array(self.points)
    self.q.put((poses, points))


class Frame(object):
  idx = 0
  last_kps = None  #last frame keypoints
  last_des = None  #last frame keypoints desctriptor
  last_pose = None  #last frame camera pose

  def __init__(self, image, calib):
    Frame.idx += 1
    self.image = image
    self.idx = Frame.idx
    self.cur_kps = None
    self.cur_des = None
    #euler transformation matrix (4*4) to represent camera pose, relative to first frame.
    self.cur_pose = None   
    self.calib = calib
    #tracking kalman filter
    self.kf = KF()

  def process(self):
    self.kf.Predict()

    #setp1, get the key points and brief
    self.cur_kps, self.cur_des = self._extract_features_v2()
    
    if Frame.idx == 1:  
      #set first frame as init, while its camera coord as world coord
      self.cur_pose = np.eye(4)
      points4 = [[0,0,0,1]]  #[0,0,0] coord, 1 is color
    else:
      match_kps = self._match_points()
      print("frame: {}, curr_des: {}, last_des: {}, match_kps: {}".
            format(Frame.idx, len(self.cur_des), len(Frame.last_des), len(match_kps)))
      #fitting essential matrix from matched points pair from last and cur frame
      essential_matrix = self._fit_essential_matrix(match_kps)

      #decompose camera pose Rt
      Rt = self._decomposeRT(essential_matrix)

      #kalman filter tracking on position
      kf_z = np.array(Rt[:3,3])
      self.kf.Update(kf_z)
      #Rt[:3,3] = self.kf.kf.x[0:3].T

      #get cur pose based on tranformation on last pose
      self.cur_pose = np.dot(Rt, Frame.last_pose)
      #calculate depth of keypoints
      points4 = self._triangulate(Frame.last_kps, self.cur_kps, 
                                  Frame.last_pose, self.cur_pose)
 
      filtered_pts4 = self._check_points(points4)
      points4 = points4[filtered_pts4]

    #plot camera poses and keypoints cloud
    mapp.add_observation(self.cur_pose, points4)

    if self.cur_pose is not None and self.last_pose is not None:
      trans = self.cur_pose[:,3] - self.last_pose[:,3]
      print('==>camera translation changes:', trans[:3])

    Frame.last_kps = self.cur_kps
    Frame.last_des = self.cur_des
    Frame.last_pose = self.cur_pose
    self.draw()

  def _check_points(self, points4):
    #check if this 3d point is in front of cameras
    idx = points4[:, 2] > 0
    return idx

  def _triangulate(self, last_kps, cur_kps, last_pose, cur_pose):
    last_kps_norm = self._normalize(last_kps)
    cur_kps_norm = self._normalize(cur_kps)

    #SVD method calculate depth
    points4 = np.zeros((last_kps.shape[0], 4))
    for i, (last_kpn,cur_kpn) in enumerate(zip(last_kps_norm, cur_kps_norm)):
      A = np.zeros((4,4))
      A[0] = last_kpn[0] * last_pose[2] - last_pose[0]
      A[1] = last_kpn[1] * last_pose[2] - last_pose[1]
      A[2] = cur_kpn[0] * cur_pose[2] - cur_pose[0]
      A[3] = cur_kpn[1] * cur_pose[2] - cur_pose[1]
      _, _, vt = np.linalg.svd(A)
      points4[i] = vt[3]
    points4 /= points4[:, 3:]  #depth normalize, [X,Y,Z,1]
    return points4



  def _decomposeRT(self, E):
    W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=np.float32)
    U,d,Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
      U *= -1.0
    if np.linalg.det(Vt) < 0:
      Vt *= -1.0
    
    R = (np.dot(np.dot(U, W), Vt))
    if np.sum(R.diagonal()) < 0:
      R = (np.dot(np.dot(U, W.T), Vt))

    t = U[:,2]
    Rt = np.eye(4)
    Rt[:3,:3] = R
    Rt[:3,3] = t
    return Rt

  def _fit_essential_matrix(self, match_kps):
    match_kps = np.array(match_kps)

    #normalize the matched points pixel
    #get [Xc/Zc, Yc/Zc]
    norm_cur_kps = self._normalize(match_kps[:,0])
    norm_last_kps = self._normalize(match_kps[:,1])

    #get essential matrix from last and cur match points
    model, _ = ransac((norm_last_kps, norm_cur_kps),
                      EssentialMatrixTransform,
                      min_samples=8,
                      residual_threshold=0.05,
                      max_trials=200)
    return model.params

  def _normalize(self, pts):
    Kinv = np.linalg.inv(self.calib.K)
    add_ones = lambda x:np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    #(K.inv)*[x,y] = [Xc/Zc, Yc/Zc, 1]
    norm_pts = np.dot(Kinv, add_ones(pts).T).T[:,0:2]
    return norm_pts

  def _extract_features_v2(self):
    orb_extractor = OrbExtractor()
    image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  #to gray scale
    kps, desc = orb_extractor.Extract(image)

    #orb_cv = cv2.ORB_create()
    #kps, des = orb_cv.compute(image, kpts)
    kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])
    desc = np.array(desc)
    return kps, desc


  def _extract_features(self):
    #TODO(): SLAM divid images to grids and find keypoints individially
    #then select FAST points in each cell
    orb = cv2.ORB_create()
    image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  #to gray scale

    #find corner points, ie. edges
    #kps, des = orb.detectAndCompute(image,None)
    pts = cv2.goodFeaturesToTrack(image, 3000, qualityLevel=0.01, minDistance=3)
    kps = [cv2.KeyPoint(x=pt[0][0],y=pt[0][1], _size=20) for pt in pts]
    kps, desc = orb.compute(image, kps)
    kps = np.array([(kp.pt[0], kp.pt[1]) for kp in kps])    
    return kps, desc

  def _match_points(self):
    bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bfmatcher.knnMatch(self.cur_des, Frame.last_des, k=2)
    print("priliminary matches len:", len(matches))
    match_kps = []
    idx1 = []
    idx2 = []
    for m,n in matches:
      #each keypoints return 2 best matches(lowest distance)
      #if first match cost smaller than second match cost, regrad as confident
      if m.distance < 0.75*n.distance:
        idx1.append(m.queryIdx)
        idx2.append(m.trainIdx)
        p1 = self.cur_kps[m.queryIdx]
        p2 = Frame.last_kps[m.trainIdx]
        match_kps.append((p1,p2))
    
    #needs at least 8 points to solve essential matrix 
    assert len(match_kps) >= 8,\
            "==> matched {} point pairs.".format(len(match_kps))

    self.cur_kps = self.cur_kps[idx1]
    self.cur_des = self.cur_des[idx1]
    Frame.last_kps = Frame.last_kps[idx2]
    Frame.last_des = Frame.last_des[idx2]
    
    return match_kps

  def draw(self):
    if self.cur_kps is not None and self.last_kps is not None:
      for kp1, kp2 in zip(self.cur_kps, Frame.last_kps):
        u1, v1 = int(kp1[0]), int(kp1[1])
        u2, v2 = int(kp2[0]), int(kp2[1])
        #current frame keypoint
        cv2.circle(self.image, (u1,v1), color=(0,0,255), radius=3)
        #last frame keypoints to current frame keypoints
        cv2.line(self.image, (u1,v1), (u2,v2), color=(255,0,0))



if __name__ == "__main__":
  dataset = 'default' #'kitti_tracking'
  mapp = Map(1024, 768)
  calib = CameraConfig(dataset)

  duration = 0 

  if dataset == 'default':
    cur_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(cur_path, 'data/road.mp4')
    cap = cv2.VideoCapture(data_path)
    while cap.isOpened():
      ret, image = cap.read()
      if ret:
        frame = Frame(image, calib)
        frame.process()
        frame.draw()
      else:
        break

      duration+=1
      print(duration)
      
      cv2.imshow('frame', frame.image)
      mapp.display()
      #cv2.waitKey(0)
      cv2.destroyAllWindows() 
  elif dataset == 'kitti_tracking':
    data_path = '/home/zuyuan/Data/kitti_tracking/raw_data/training/image_2/0000'
    images = os.listdir(data_path)
    images = sorted(images)
    for i in range(len(images)):
      img_name = images[i]
      img_name = os.path.join(data_path, img_name)
      img = cv2.imread(img_name) 

      frame = Frame(img, calib)
      frame.process()
      frame.draw()

      duration+=1
      print(duration)

      cv2.imshow('frame', frame.image)
      mapp.display()
      #cv2.waitKey(0)
      cv2.destroyAllWindows() 

  else:
    raise TypeError("dataset not supported")
