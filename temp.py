import SimpleITK as sitk
import os
import numpy as np

file_name = "./subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"
itk_img = sitk.ReadImage(file_name)
img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
# center = np.array([node_x, node_y, node_z])   # nodule center
origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
# v_center =np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering

print(spacing)