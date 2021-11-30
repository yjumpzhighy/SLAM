import numpy as np
import cv2
import math

from .pattern import bit_pattern_31


factorPI = (float)(np.pi/180.)
class OrbExtractor(object):
  #ORB检测Oriented FAST关键点时选取的图像块边长，
  #即计算质心时选取的图像块区域边长
  HALF_PATCH_SIZE = 15
  PATCH_SIZE = 31
  EDGE_THRESHOLD = 19
  W = 30 #grid size, unit pixel
  def __init__(self):
    self.nfeatures = 1000   #total keypoints extracted
    self.nlevels = 5  #pyramid levels
    self.scaleFactor = 0.8  #[0,1]
    self.initThFAST = 20  #threshold for greyscale value between P and neighbors in FAST
    self.minThFast = 7
    self.ScaleFactors = np.zeros((self.nlevels,))
    self.LevelSigma = np.zeros((self.nlevels,))
    self.ScaleFactors[0] = 1.
    self.LevelSigma[0] = 1.
    for i in range(1, self.nlevels):
      self.ScaleFactors[i] = self.ScaleFactors[i-1] * self.scaleFactor
      self.LevelSigma[i] = self.ScaleFactors[i] * self.ScaleFactors[i]
    self.InvScaleFactors = np.zeros((self.nlevels,))
    self.InvLevelSigma = np.zeros((self.nlevels,))
    for i in range(1, self.nlevels):
      self.InvScaleFactors[i] = 1. / self.ScaleFactors[i]
      self.InvLevelSigma[i] = 1. / self.LevelSigma[i]

    self.ImagePyramid = []
    self.FeaturePerLevel = np.zeros((self.nlevels,))
    #factor = 1. / self.scaleFactor
    factor = self.scaleFactor * self.scaleFactor
    DesiredFeaturePerScale = self.nfeatures*(1-factor)/(1-pow(factor, self.nlevels))
    
    sum_features = 0
    for i in range(self.nlevels - 1):
      self.FeaturePerLevel[i] = np.round(DesiredFeaturePerScale)
      sum_features += self.FeaturePerLevel[i]
      DesiredFeaturePerScale *= factor
    self.FeaturePerLevel[self.nlevels-1] = max(self.nfeatures-sum_features, 0)

    #用于后面计算描述子的随机采样点集合
    #最后通过求 x 坐标对应在半径为 HALF_PATCH_SIZE（15, 使用灰度质心法计算特征点的方向信息时，
    # 图像块的半径）的圆上的 y 坐标，标出了一个圆形区域用来求特征点方向；
    npoints = 512
    self.pattern = []
    for i in range(npoints):
      self.pattern.append((bit_pattern_31[i*2], bit_pattern_31[i*2+1]))

    #This is for orientation
    #表示为1/4圆的弦上, v点对应的最大横坐标u
    #注意45度以下计算得到，45度以上按照对称得到
    self.umax = np.zeros((self.HALF_PATCH_SIZE+1,),dtype=np.int32)
    vmax = int(np.floor(self.HALF_PATCH_SIZE * np.sqrt(2.) / 2 + 1))
    vmin = int(np.ceil(self.HALF_PATCH_SIZE * np.sqrt(2.) / 2))
    hp2 = self.HALF_PATCH_SIZE * self.HALF_PATCH_SIZE
    for v in range(vmax+1):
      self.umax[v] = np.round(np.sqrt(hp2 - v*v))

    #使用了对称的方式计算上四分之一的圆周上的umax，目的也是为了保持严格的对称（如
    #果按照常规的想法做，由于cvRound就会很容易出现不对称的情况，同时这些随机采样
    #的特征点集也不能够满足旋转之后的采样不变性了）
    v0 = 0
    for v in reversed(range(vmin, self.HALF_PATCH_SIZE+1)):
      while self.umax[v0]==self.umax[v0 + 1]:
        v0+=1
      self.umax[v] = v0
      v0+=1


  def Extract(self, img):
    #compute orb features and descriptors on an image
    #orb are dispersed on the image using octree
    if img is None:
      return

    #get image for each pyramid level
    self._ComputePyramid(img)
    #提取FAST角点和利用OctTree
    allKeypoints = self._ComputeKeyPointsOctTree()

    #计算描述子
    nKeypoints = 0
    for level in range(self.nlevels):
      nKeypoints += int(len(allKeypoints[level]))
    print("total keypoints number:", nKeypoints)

    if nKeypoints==0:
      return
    
    #描述矩阵共有nkeypoints行，32列，元素数据类型为CV_8U
    #也就是说每个特征点由32个CV_8U的数字描述
    #descriptors = np.zeros((nKeypoints, 32), dtype=np.uint8)
    offset = 0
    output_kpts = []
    output_desc = []
    #按层数依次遍历特征点
    for level in range(self.nlevels):
      keypoints = allKeypoints[level]
      nkeypointslevel = len(keypoints)
      if nkeypointslevel == 0:
        continue
 
      blurimg = np.copy(self.ImagePyramid[level])
      #ORB BRIEF对噪音敏感，高斯模糊缓解噪声敏感问题
      #第一个参数是输入，第二个参数是高斯卷积核大小
      #第三、第四个参数是X、Y方向上的方差sigma
      #第五个参数是扩边方式
      blurimg = cv2.GaussianBlur(blurimg, (7,7), 2, 2, cv2.BORDER_REFLECT_101)

      #计算特征点描述子
      #desc = descriptors[offset:offset+nkeypointslevel, :]
      desc = self._ComputeDescriptors(blurimg, keypoints, self.pattern)

      #恢复尺度后输出特征点结果
      levelscale = self.ScaleFactors[level]
      for i in range(nkeypointslevel):
        keypoints[i].pt = (keypoints[i].pt[0]/levelscale, keypoints[i].pt[1]/levelscale) 
        output_kpts.append(keypoints[i])
        output_desc.append(desc[i])
    return output_kpts, output_desc

  def _ComputeDescriptors(self, blurimg, keypoints, pattern):
    #重新确认清零
    desc = np.zeros((len(keypoints), 32), dtype=np.uint8)
    for i in range(len(keypoints)):
      desc[i,:] = self._ComputeOrbDescriptor(keypoints[i], blurimg, pattern)
    return desc

  def _ComputeOrbDescriptor(self, kpt, img, pattern):
    """
    利用FAST特征点检测时求取的主方向，旋转特征点邻域，但旋转整个Patch再提取
    BRIEF特征描述子的计算代价较大，因此，ORB采用了一种更高效的方式，在每个特征
    点邻域Patch内先选取256对(512个)随机点，将其进行旋转，然后做判决编码为二进制串
    pattern: [512, 2]
    """
    angle = float(kpt.angle) * factorPI  #degree to rad
    a = float(math.cos(angle))
    b = float(math.sin(angle))
    center_x = int(kpt.pt[0])
    center_y = int(kpt.pt[1])
    
    def GetValue(idx):
      return img[center_y+int(pattern[idx][0]*b + pattern[idx][1]*a)]\
                [center_x+int(pattern[idx][0]*a - pattern[idx][1]*b)]

    
    #一次迭代读取pattern的步长，在一次迭代中，一共会获取16个元素
    #可以计算出一共要有32×16个点用于迭代，而在代码一开始的pattern中就是512个点
    pattern_step = 16  
    desc = np.zeros((32,), dtype=np.uint8)
    for i in range(32):
      #即将16个点的对比信息压缩到一个uint里，其中每一位都代表一个点对的对比结果.
      #则每个特征点最后得到32个描述值
      t0, t1 = GetValue(i * pattern_step + 0), GetValue(i * pattern_step + 1)
      val = np.uint8(t0<t1) 
      t0, t1 = GetValue(i * pattern_step + 2), GetValue(i * pattern_step + 3)
      val |= ((t0<t1)<<1)
      t0, t1 = GetValue(i * pattern_step + 4), GetValue(i * pattern_step + 5)
      val |= ((t0 < t1) << 2)
      t0, t1 = GetValue(i * pattern_step + 6), GetValue(i * pattern_step + 7)
      val |= ((t0 < t1) << 3)
      t0, t1 = GetValue(i * pattern_step + 8), GetValue(i * pattern_step + 9)
      val |= ((t0 < t1) << 4)
      t0, t1 = GetValue(i * pattern_step + 10), GetValue(i * pattern_step + 11)
      val |= ((t0 < t1) << 5)
      t0, t1 = GetValue(i * pattern_step + 12), GetValue(i * pattern_step + 13)
      val |= ((t0 < t1) << 6)
      t0, t1 = GetValue(i * pattern_step + 14), GetValue(i * pattern_step + 15)
      val |= ((t0 < t1) << 7)

      desc[i] = val

    return desc
      



  def _ComputeKeyPointsOctTree(self):
    #将image分为W×W格子，在每个格子上单独做检测
    allKeypoints = []
    #extract features of each pyramid level
    for level in range(self.nlevels):
      feat_height, feat_width = self.ImagePyramid[level].shape
      minBorderX = self.EDGE_THRESHOLD - 3
      minBorderY = minBorderX
      maxBorderX = feat_width - self.EDGE_THRESHOLD + 3
      maxBorderY = feat_height - self.EDGE_THRESHOLD + 3
      #大量数据进行处理的时候就要使用reserve主动分配内存以提升程序执行效率
      #这里保留的是用户指定总特征个数的10倍的内存
      vToDistributeKeysPerLevel = []
      width = maxBorderX - minBorderX
      height = maxBorderY - minBorderY
      nCols = 1 if width<self.W else int(width / self.W)  #格网列数
      nRows = 1 if height<self.W else int(height / self.W)  #格网行数
      wCell = int(np.ceil(width/nCols)) #每个格网的宽度，向上取整
      hCell = int(np.ceil(height/nRows))

      #依照行列遍历每个格网进行处理
      #这一部分其实可以考虑用GPU进行并行加速，因为每个格网提取FAST角点的过程是各自独立的
      for i in range(nRows):
        iniY = minBorderY + i*hCell
        if iniY >= maxBorderY-3:
          continue 
        maxY = iniY + hCell + 6
        if maxY > maxBorderY:
          maxY = maxBorderY
        
        for j in range(nCols):
          iniX = minBorderX + j*wCell
          if iniX >= maxBorderX-6:
            continue
          maxX = iniX + wCell + 6
          if maxX > maxBorderX:
            maxX = maxBorderX

          #每一个格网都会建一个临时的vector用于存放这个小格子里的特征点
          fast_detector = cv2.FastFeatureDetector_create(threshold=self.initThFAST,
                                                         nonmaxSuppression=True,
                                                         type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
          vKeysCell = fast_detector.detect(self.ImagePyramid[level][iniY:maxY, iniX:maxX])
         
          #print("==>Init FAST didn't detect keypoints, use min Fast instead.")
          if not vKeysCell:
            fast_detector = cv2.FastFeatureDetector_create(threshold=self.minThFast,
                                                           nonmaxSuppression=True,
                                                           type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
            vKeysCell = fast_detector.detect(self.ImagePyramid[level][iniY:maxY, iniX:maxX])

          
          #对于每一小块grid，如果提取的特征点不为空进行后续处理
          if vKeysCell:
            maxresponse = vKeysCell[0].response
            idx = 0
            for k in range(1, len(vKeysCell)):
              if vKeysCell[k].response > maxresponse:
                maxresponse = vKeysCell[k].response
                idx = k

            keypnt = vKeysCell[idx]
            keypnt.pt = (keypnt.pt[0]+j*wCell, keypnt.pt[1]+i*hCell)

            if len(vToDistributeKeysPerLevel) < self.FeaturePerLevel[level]:
              vToDistributeKeysPerLevel.append(keypnt)

      #print("level:".format(level), len(vToDistributeKeysPerLevel))
      #利用八叉树对每一层提取的特征点进行过滤，使其均匀分布
      #提取keypoints时已经直接简单完成这一步,skip

      scaledPatchSize = self.PATCH_SIZE * self.ScaleFactors[level]

      for keypnt in vToDistributeKeysPerLevel:
        #其实就是相当于整体平移一下
        keypnt.pt = (keypnt.pt[0]+minBorderX, keypnt.pt[1]+minBorderY)
        #octave是OpenCV的KeyPoint类的属性之一，int类型，说明这个特征点是位于金字塔的哪一层
        keypnt.octave = level
        #size也是KeyPoint的属性之一，float类型，说明该特征点的有效范围(直径)
        #FAST本身并没有严格意义上的有效范围概念(硬要说也有，就是中心点与周围比较像素构成的圆的直径)，
        #但ORB用的是Oriented FAST，在计算方向时有用到PATCH_SIZE，因此在这里就把计算方向的范围
        #作为特征点的有效范围.根据上面的计算公式也知道，scaledPatchSize与PATCH_SIZE和不同层的缩放
        #因子有关.在PATCH_SIZE一定的情况下(例如这里在代码一开始就设置为了30)只与缩放因子有关，
        #每一层的特征点有效范围都一样
        keypnt.size = scaledPatchSize
      allKeypoints.append(vToDistributeKeysPerLevel) 

    #计算角点的方向，也是特征点提取的倒数第二步(最后一步是计算描述子)
    for level in range(self.nlevels):
      self._ComputeOrientation(self.ImagePyramid[level], allKeypoints[level], self.umax)

    return allKeypoints


  def _ComputeOrientation(self, img, keypnts, umax):
    """
    计算每个keypnt的质心方向
    img: image of this pyramid level
    keypnts: detected fast keypoints of this pyramid level
    umax: u坐标绝对值的最大值
    """
    for kpt in keypnts:
      kpt.angle = self._IC_Angle(img, kpt.pt, umax)
  
  def _IC_Angle(self, img, pt, umax):
    """
    灰度质心法：以几何中心和灰度质心的连线作为该特征点方向
    img: image of this pyramid level
    pt: point (u,v) in pixel units
    umax: u坐标绝对值的最大值
    """
    m_01 = 0
    m_10 = 1
    center_x = int(pt[0])
    center_y = int(pt[1])

    #先单独算出中间v=0这一行
    for u in range(-self.HALF_PATCH_SIZE, self.HALF_PATCH_SIZE+1):
      m_10 += u * img[center_y][center_x-u]

    #本来m_01应该是一列一列地计算的，但是由于对称以及坐标x,y正负的原因，可以一次计算两行
    for v in range(1, self.HALF_PATCH_SIZE+1):
      v_sum = 0
      w_max =self.umax[v]  #某行像素横坐标的最大范围，注意这里的图像块是圆形的
      #在坐标范围内挨个像素遍历，实际是一次遍历2个
      #假设每次处理的两个点坐标，中心线上方为(x,y),中心线下方为(x,-y) 
      #对于某次待处理的两个点：m_10 = Σ x*I(x,y) =  x*I(x,y) + x*I(x,-y) = x*(I(x,y) + I(x,-y))
      #对于某次待处理的两个点：m_01 = Σ y*I(x,y) =  y*I(x,y) - y*I(x,-y) = y*(I(x,y) - I(x,-y))
      for u in range(-w_max, w_max+1):
        val_plus = int(img[center_y+v][center_x+u])
        val_minus = int(img[center_y-v][center_x+u])
        v_sum += (val_plus - val_minus)
        m_10 += u * (val_minus + val_plus)
      
      m_01 += v*v_sum
    return cv2.fastAtan2(float(m_01), float(m_10))


  def _ComputePyramid(self, img):
    for level in range(self.nlevels):
      scale = self.ScaleFactors[level]
      #根据尺寸因子计算长宽尺寸
      feat_width = int(np.round(float(img.shape[1]) * scale))
      feat_height = int(np.round(float(img.shape[0]) * scale))      
      #计算每层图像
      if level != 0:
        temp = cv2.resize(self.ImagePyramid[level-1],
                          (feat_width, feat_height), 
                          interpolation = cv2.INTER_LINEAR)
        temp = cv2.copyMakeBorder(temp, self.EDGE_THRESHOLD,
                                  self.EDGE_THRESHOLD, self.EDGE_THRESHOLD,
                                  self.EDGE_THRESHOLD, cv2.BORDER_REFLECT_101)
      else:
        temp = cv2.copyMakeBorder(img, self.EDGE_THRESHOLD,
                                  self.EDGE_THRESHOLD, self.EDGE_THRESHOLD,
                                  self.EDGE_THRESHOLD, cv2.BORDER_REFLECT_101)
      self.ImagePyramid.append(temp[int(self.EDGE_THRESHOLD) : int(self.EDGE_THRESHOLD+feat_height),
                                    int(self.EDGE_THRESHOLD) : int(self.EDGE_THRESHOLD+feat_width)])          

def Match(cur_kps, last_kps, cur_des, last_des):
    bfmatcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bfmatcher.knnMatch(cur_des, last_des, k=2)
    match_kps = []
    for m,n in matches:
      #each keypoints return 2 best matches(lowest distance)
      #if first match cost smaller than second match cost, regrad as confident
      if m.distance < 0.75*n.distance:
        match_kps.append(m)
    
    #needs at least 8 points to solve essential matrix 
    assert len(match_kps) >= 8,\
            "==> matched {} point pairs.".format(len(match_kps))   
    return match_kps

    
if __name__ == "__main__":
  img_name = '/home/zuyuan/Data/kitti_tracking/raw_data/training/image_2/0000/000003.png'
  img_raw = cv2.imread(img_name) 
  img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)  #to gray scale

  #open-cv build-in lib function:
  orb = cv2.ORB_create()
  pts = cv2.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=3)
  kpts = [cv2.KeyPoint(x=pt[0][0],y=pt[0][1], _size=20) for pt in pts]
  for kpt in kpts:
    cv2.circle(img_raw, (int(kpt.pt[0]),int(kpt.pt[1])), 
               2, (0,0,255), thickness=1, lineType=8, shift=0) #red

  #improved from orb-slam2:
  orb = OrbExtractor()
  kpts, desc = orb.Extract(img)
  for kpt in kpts:
    cv2.circle(img_raw, (int(kpt.pt[0]),int(kpt.pt[1])), 
               2, (0,255,0), thickness=1, lineType=8, shift=0) #blue
  kpts_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kpts])
  desc_np = np.array(desc)
  
  img2_name = '/home/zuyuan/Data/kitti_tracking/raw_data/training/image_2/0000/000004.png'
  img2_raw = cv2.imread(img2_name) 
  img2 = cv2.cvtColor(img2_raw, cv2.COLOR_BGR2GRAY)  #to gray scale
  kpts2, desc2 = orb.Extract(img2)
  kpts2_np = np.array([(kp.pt[0], kp.pt[1]) for kp in kpts2])
  desc2_np = np.array(desc2)
  match = Match(kpts2_np, kpts_np, desc2_np, desc_np)
  print(match)
  match_imgs = cv2.drawMatches(img2_raw, kpts2, img_raw, kpts, match, outImg=None)
  cv2.imshow("Match Result", match_imgs)

  cv2.imshow('compare', img_raw)
  cv2.waitKey(0)
  cv2.destroyAllWindows() 
  
