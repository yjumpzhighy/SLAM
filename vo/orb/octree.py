import numpy as np
import open3d as o3d
import open3d_tutorial as o3dtut

 
# # 加载点云，并采样2000个点
# N = 2000
# pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
# # 点云归一化
# pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
#           center=pcd.get_center())
# # 点云着色
# pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3)))
# # 可视化
# #o3d.visualization.draw_geometries([pcd])
# print(np.asarray(pcd.points))

# # 创建八叉树， 树深为4
# octree = o3d.geometry.Octree(max_depth=4)
# # 从点云中构建八叉树，适当扩展边界0.01m
# octree.convert_from_point_cloud(pcd, size_expand=0.01)
# # 可视化
# o3d.visualization.draw_geometries([octree])

xyz = np.array([[1,2,0],[4,5,0],[2,6,0],[3,4,0],
                [1,3,0],[9,5,0],[2,3,0],[6,7,0]])
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(8, 3)))

octree = o3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(pcd)
print(type(octree))

o3d.visualization.draw_geometries([octree])
