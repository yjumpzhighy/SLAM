# Stereo Camera
Stereo Calculate Depth
Four steps for stereo images process:
1. images rectify
2. cost calculation (DSI) with epipolor constraints.
3. cost aggregation
4. desparity calculation
5. desparity optimization
6. depth estimation

Results:
![alt text](https://github.com/yangzuyuanhao/VO/blob/c5ebe4130eebf5f890ef812a03224eff0630891e/stereo/rectified.png)

![alt text](https://github.com/yangzuyuanhao/VO/blob/c5ebe4130eebf5f890ef812a03224eff0630891e/stereo/cloud.png
)


# VO
Mono-camera Visual Odometer
1. Orb features extraction
   (adopted orb-slam2 method, evenly distribute keypoints across image)
   
   
   
2. Keypoints match with last frame
3. Build essential matrix E from matched point pairs (>8)
4. Decompose R,T from essential matrix
5. Kalman filter to filter R,T
6. Apply R,T based on last frame camera pose
7. Get camera pose of this frame



# VIN
