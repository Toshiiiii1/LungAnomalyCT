import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os

file_list = os.listdir("./subset0")
file_list = list(map(lambda file : "./subset0/" + file, file_list))

def get_filename(case):
    global file_list
    for f in file_list:
        if case in f:
            return(f)
        
def matrix2int16(matrix):
    ''' 
        matrix must be a numpy array NXN
        Returns uint16 version
    '''
    m_min= np.min(matrix)
    m_max= np.max(matrix)
    matrix = matrix-m_min
    return(np.array(np.rint( (matrix-m_min)/float(m_max-m_min) * 65535.0), dtype=np.uint16))

if __name__ == "__main__":
    df_node = pd.read_csv("./annotations.csv")
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()
    
    fcount = 0
    for img_file in file_list:
        # print("Getting mask for image file %s" % img_file.replace("./subset0",""))
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if len(mini_df) > 0:       # some files may not have a nodule--skipping those
            itk_img = sitk.ReadImage(img_file) # read .mhd file
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            numz, height, width = img_array.shape
            center = np.array([node_x, node_y, node_z])   # nodule center
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering
            
            i = 0
            for i_z in range(int(v_center[2])-1, int(v_center[2])+2):
                mask = make_mask(center, diam, i_z*spacing[2]+origin[2], width, height, spacing, origin)
                masks[i] = mask
                imgs[i] = matrix2int16(img_array[i_z])
                i+=1
            np.save(output_path+"images_%d.npy" % (fcount) ,imgs)
            np.save(output_path+"masks_%d.npy" % (fcount) ,masks)