import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import os
from tqdm import tqdm

file_list = os.listdir("./subset0")
file_list = list(map(lambda file : "./subset0/" + file, file_list))

def make_mask(center, diam, z, width, height, spacing, origin):
    '''
        Center : centers of circles px -- list of coordinates x,y,z
        diam : diameters of circles px -- diameter
        widthXheight : pixel dim of image
        spacing = mm/px conversion rate np array x,y,z
        origin = x,y,z mm np.array
        z = z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width]) # 0's everywhere except nodule swapping x,y to match img
    #convert to nodule space from world coordinates

    # Defining the voxel range in which the nodule falls
    v_center = (center - origin)/spacing
    v_diam = int(diam/spacing[0] + 1)
    v_xmin = np.max([0, int(v_center[0]-v_diam) - 1])
    v_xmax = np.min([width-1, int(v_center[0]+v_diam) + 1])
    v_ymin = np.max([0, int(v_center[1]-v_diam) - 1]) 
    v_ymax = np.min([height-1, int(v_center[1]+v_diam) + 1])

    v_xrange = range(v_xmin, v_xmax+1)
    v_yrange = range(v_ymin, v_ymax+1)

    # Convert back to world coordinates for distance calculation
    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y-origin[1]) / spacing[1]), int((p_x-origin[0]) / spacing[0])] = 1.0
    return(mask, [v_xmin, v_xmax, v_ymin, v_ymax])

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
    df_roi = pd.read_csv("./ROI_coor.csv")
    df_roi_cur = dict()
    
    for fcount, img_file in enumerate(tqdm(file_list)):
        # print("Getting mask for image file %s" % img_file.replace("./subset0",""))
        mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
        if len(mini_df) > 0:       # some files may not have a nodule--skipping those
            itk_img = sitk.ReadImage(img_file) # read .mhd file
            img_array = sitk.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
            num_z, height, width = img_array.shape
            origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
            spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
            
            for node_idx, cur_row in mini_df.iterrows():       
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                diam = cur_row["diameter_mm"]
                
                imgs = np.ndarray([5, height, width], dtype=np.float32)
                masks = np.ndarray([5, height, width], dtype=np.uint8)
                center = np.array([node_x, node_y, node_z])   # nodule center
                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
            
                for i, i_z in enumerate(np.arange(int(v_center[2])-2, int(v_center[2])+3).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                    mask, roi = make_mask(center, diam, i_z*spacing[2]+origin[2], width, height, spacing, origin)
                    if i == 2:
                        # roi_key = "roi_%04d_%04d" % (fcount, node_idx)
                        roi_key = f"{file_list[0].split('/')[-1].rsplit('.', 1)[0]}_{fcount}_{node_idx}"
                        df_roi_cur.update({roi_key: roi})
                    masks[i] = mask
                    imgs[i] = img_array[i_z]
                img_name = f"{file_list[0].split('/')[-1].rsplit('.', 1)[0]}_{fcount}_{node_idx}.npy"
                mask_name = f"{file_list[0].split('/')[-1].rsplit('.', 1)[0]}_{fcount}_{node_idx}.npy"
                
                np.save(os.path.join("./images", img_name), imgs)
                np.save(os.path.join("./masks", mask_name), masks)
            
    df_temp = pd.DataFrame.from_dict(df_roi_cur, columns=["x_min", "x_max", "y_min", "y_max"], orient='index')
    df_temp["ID"] = list(df_roi_cur.keys())
    df_temp = df_temp.reset_index(drop=True)
    df_roi = pd.concat([df_roi, df_temp])
    df_roi.to_csv("ROI_coor.csv", index=False)
        