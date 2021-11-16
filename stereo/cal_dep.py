# -*- coding: utf-8 -*-
import cv2
import numpy as np
import stereoconfig
import pcl
import pcl.pcl_visualization
 

def preprocess(img1, img2):
    # col image->grayscale
    if(img1.ndim == 3):
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # OpenCV load image as BGR
    if(img2.ndim == 3):
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
 
    # histogram balance
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)
 
    return img1, img2
 
 
# remove distort
def undistortion(image, camera_matrix, dist_coeff):
    undistortion_image = cv2.undistort(image, camera_matrix, dist_coeff)
 
    return undistortion_image
 
 
# get rotation, projection, unprojection matrix
def getRectifyTransform(height, width, config):
    left_K = config.cam_matrix_left
    right_K = config.cam_matrix_right
    left_distortion = config.distortion_l
    right_distortion = config.distortion_r
    R = config.R
    T = config.T
 
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(left_K, left_distortion, right_K, right_distortion, (width, height), R, T, alpha=0)
 
    map1x, map1y = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)
 
    return map1x, map1y, map2x, map2y, Q
 
 
# rectify images
def rectifyImage(image1, image2, map1x, map1y, map2x, map2y):
    rectifyed_img1 = cv2.remap(image1, map1x, map1y, cv2.INTER_AREA)
    rectifyed_img2 = cv2.remap(image2, map2x, map2y, cv2.INTER_AREA)
 
    return rectifyed_img1, rectifyed_img2
 
 
# verify rectify result 
def draw_line(image1, image2):
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
 
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:image1.shape[0], 0:image1.shape[1]] = image1
    output[0:image2.shape[0], image1.shape[1]:] = image2
 
    # draw parallel lines
    line_interval = 50  # space：50
    for k in range(height // line_interval):
        cv2.line(output, (0, line_interval * (k + 1)), (2 * width, line_interval * (k + 1)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
 
    output = cv2.resize(output, (int(width/4), int(height/4)))
    return output
 
 
# disparity calculation
def stereoMatchSGBM(left_image, right_image, down_scale=False):
    if left_image.ndim == 2:
        img_channels = 1
    else:
        img_channels = 3
    blockSize = 3
    paraml = {'minDisparity': 0,
             'numDisparities': 128,
             'blockSize': blockSize,
             'P1': 8 * img_channels * blockSize ** 2,
             'P2': 32 * img_channels * blockSize ** 2,
             'disp12MaxDiff': 1,
             'preFilterCap': 63,
             'uniquenessRatio': 15,
             'speckleWindowSize': 100,
             'speckleRange': 1,
             'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
             }
 
    left_matcher = cv2.StereoSGBM_create(**paraml)
    paramr = paraml
    paramr['minDisparity'] = -paraml['numDisparities']
    right_matcher = cv2.StereoSGBM_create(**paramr)
 
    size = (left_image.shape[1], left_image.shape[0])
    if down_scale == False:
        disparity_left = left_matcher.compute(left_image, right_image)
        disparity_right = right_matcher.compute(right_image, left_image)
 
    else:
        left_image_down = cv2.pyrDown(left_image)
        right_image_down = cv2.pyrDown(right_image)
        factor = left_image.shape[1] / left_image_down.shape[1]
 
        disparity_left_half = left_matcher.compute(left_image_down, right_image_down)
        disparity_right_half = right_matcher.compute(right_image_down, left_image_down)
        disparity_left = cv2.resize(disparity_left_half, size, interpolation=cv2.INTER_AREA)
        disparity_right = cv2.resize(disparity_right_half, size, interpolation=cv2.INTER_AREA)
        disparity_left = factor * disparity_left
        disparity_right = factor * disparity_right
 
    # SGBM calculated disparity scales 16, needs to scale back.
    trueDisp_left = disparity_left.astype(np.float32) / 16.
    trueDisp_right = disparity_right.astype(np.float32) / 16.
 
    return trueDisp_left, trueDisp_right
 
 
def hw3ToN3(points):
    height, width = points.shape[0:2]
 
    points_1 = points[:, :, 0].reshape(height * width, 1)
    points_2 = points[:, :, 1].reshape(height * width, 1)
    points_3 = points[:, :, 2].reshape(height * width, 1)
 
    points_ = np.hstack((points_1, points_2, points_3))
 
    return points_
 
 
# convert rgbd to point clouds
def DepthColor2Cloud(points_3d, colors):
    rows, cols = points_3d.shape[0:2]
    size = rows * cols
 
    points_ = hw3ToN3(points_3d)
    colors_ = hw3ToN3(colors).astype(np.int64)
 
    blue = colors_[:, 0].reshape(size, 1)
    green = colors_[:, 1].reshape(size, 1)
    red = colors_[:, 2].reshape(size, 1)
 
    rgb = np.left_shift(blue, 0) + np.left_shift(green, 8) + np.left_shift(red, 16)
 
    pointcloud = np.hstack((points_, rgb)).astype(np.float32)
 
    # bound
    X = pointcloud[:, 0]
    Y = pointcloud[:, 1]
    Z = pointcloud[:, 2]
 
    remove_idx1 = np.where(Z <= 0)
    remove_idx2 = np.where(Z > 15000)
    remove_idx3 = np.where(X > 10000)
    remove_idx4 = np.where(X < -10000)
    remove_idx5 = np.where(Y > 10000)
    remove_idx6 = np.where(Y < -10000)
    remove_idx = np.hstack((remove_idx1[0], remove_idx2[0], remove_idx3[0], remove_idx4[0], remove_idx5[0], remove_idx6[0]))
 
    pointcloud_1 = np.delete(pointcloud, remove_idx, 0)
 
    return pointcloud_1
 
 
def view_cloud(pointcloud):
    cloud = pcl.PointCloud_PointXYZRGBA()
    cloud.from_array(pointcloud)
 
    try:
        visual = pcl.pcl_visualization.CloudViewing()
        visual.ShowColorACloud(cloud)
        v = True
        while v:
            v = not (visual.WasStopped())
    except:
        pass
 
 
if __name__ == '__main__':
    iml = cv2.imread('/home/Data/middlebury/stereo_im/im0.png')  
    imr = cv2.imread('/home/Data/middlebury/stereo_im/im1.png')
    height, width = iml.shape[0:2]
    
    config = stereoconfig.stereoCamera()
 
    map1x, map1y, map2x, map2y, Q = getRectifyTransform(height, width, config)  
    iml_rectified, imr_rectified = rectifyImage(iml, imr, map1x, map1y, map2x, map2y)
 
    if True:
        line = draw_line(iml, imr)
        line_rectified = draw_line(iml_rectified, imr_rectified)
        cv2.imshow('image window raw', line)
        cv2.imshow('image window rect', line_rectified)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   


    iml_, imr_ = preprocess(iml, imr) 
    disp, _ = stereoMatchSGBM(iml_, imr_, True) 
    if False:
        dh, dw = disp.shape[0:2]
        disp_show = cv2.resize(disp, (int(dw/4), int(dh/4)))
        cv2.imshow('image disp', disp_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()   
 ）
    points_3d = cv2.reprojectImageTo3D(disp, Q)  
    pointcloud = DepthColor2Cloud(points_3d, iml)
 
    view_cloud(pointcloud)
