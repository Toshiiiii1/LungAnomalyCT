import requests
from tqdm import tqdm
import argparse
import zipfile
import os
import SimpleITK as sitk
import numpy as np
import csv
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def download_file(url, output_path, chunk_size=1024*1024):
    """
    Download files from LUNA16 dataset, including .zip and .csv
    
    Args:
        url: url path of file
        output_path: output path
        chunk_size: the size of each chunk
    """
    print(f"Downloading {url}...")
    
    # checking whether url work or not
    res = requests.get(url, stream=True)
    res.raise_for_status()

    # get total size
    total_size = int(res.headers.get("content-length", 0))
    
    if total_size == 0:
        with open(output_path, "wb") as file:
            file.write(res.content)

    with open(output_path, "wb") as file, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024
    ) as bar:
        # split data into chunks, download each chunk
        for chunk in res.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

    print(f"{output_path} was downloaded successfully")
    
    # Auto extract if it's a zip file
    if output_path.endswith(".zip"):
        print(f"Extracting {output_path}...")
        root = os.path.dirname(output_path)
        with zipfile.ZipFile(output_path, "r") as zip_file:
            zip_file.extractall(root)
        print(f"Extracted to {root}")
        
        # Remove zip file after extraction
        os.remove(output_path)
        print(f"Removed {output_path}")

def make_mask(center, diam, z, width, height, spacing, origin):
    '''
    Create mask for nodule detection
    
    Args:
        center: centers of circles px -- list of coordinates x,y,z
        diam: diameters of circles px -- diameter
        width, height: pixel dimensions of image
        spacing: mm/px conversion rate np array x,y,z
        origin: x,y,z mm np.array
        z: z position of slice in world coordinates mm
    '''
    mask = np.zeros([height, width])
    
    # Convert to nodule space from world coordinates
    v_center = (center - origin) / spacing
    v_diam = int(diam / spacing[0] + 1)
    v_xmin = np.max([0, int(v_center[0] - v_diam/2) - 2])
    v_xmax = np.min([width-1, int(v_center[0] + v_diam/2) + 2])
    v_ymin = np.max([0, int(v_center[1] - v_diam/2) - 2]) 
    v_ymax = np.min([height-1, int(v_center[1] + v_diam/2) + 2])

    v_xrange = range(v_xmin, v_xmax+1)
    v_yrange = range(v_ymin, v_ymax+1)

    # Fill in 1 within sphere around nodule
    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                mask[int((p_y-origin[1]) / spacing[1]), int((p_x-origin[0]) / spacing[0])] = 1.0
                
    return mask, [v_xmin, v_xmax, v_ymin, v_ymax]

def get_filename(case, file_list):
    """Find corresponding .mhd file for given series UID"""
    for f in file_list:
        if case in f:
            return f
    return None
        
def normalize(image):
    """Normalize CT image using standard windowing"""
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def process_ct_data(subset_name, data_dir, output_dir, clean_output=False, separate_subsets=False):
    """
    Process CT data to extract nodule images and ROI coordinates
    
    Args:
        subset_name: name of the subset (e.g., 'subset0')
        data_dir: directory containing the extracted data
        output_dir: directory to save processed results
        clean_output: whether to clean output directory before processing
        separate_subsets: whether to create separate subdirectories for each subset
    """
    print(f"Processing {subset_name}...")
    
    # Handle output directory structure
    if separate_subsets:
        # Create separate directory for each subset: output_dir/subset0/, output_dir/subset1/
        current_output_dir = os.path.join(output_dir, subset_name)
        ct_images_dir = os.path.join(current_output_dir, "ct_images")
        output_csv_path = os.path.join(current_output_dir, f"{subset_name}_output.csv")
    else:
        # Use common directory: output_dir/ct_images/, output_dir/output.csv
        current_output_dir = output_dir
        ct_images_dir = os.path.join(output_dir, "ct_images")
        output_csv_path = os.path.join(output_dir, "output.csv")
    
    # Clean output directory if requested
    if clean_output and os.path.exists(ct_images_dir):
        print(f"Cleaning existing images in {ct_images_dir}")
        import shutil
        shutil.rmtree(ct_images_dir)
    
    # Create output directories
    os.makedirs(ct_images_dir, exist_ok=True)
    os.makedirs(current_output_dir, exist_ok=True)
    
    # Get list of .mhd files
    subset_dir = os.path.join(data_dir, subset_name)
    file_list = glob(os.path.join(subset_dir, "*.mhd"))
    
    if not file_list:
        print(f"No .mhd files found in {subset_dir}")
        return
    
    print(f"Found {len(file_list)} .mhd files")
    
    # Load annotations
    annotations_path = os.path.join(data_dir, "annotations.csv")
    if not os.path.exists(annotations_path):
        print(f"Annotations file not found: {annotations_path}")
        return
        
    nodule_annotations = pd.read_csv(annotations_path)
    nodule_annotations["file"] = nodule_annotations["seriesuid"].apply(lambda x: get_filename(x, file_list))
    nodule_annotations = nodule_annotations.dropna()
    
    # Initialize output CSV
    columns = ['x_min', 'x_max', 'y_min', 'y_max', 'ID']
    
    # Create CSV with headers only if it doesn't exist
    if not os.path.exists(output_csv_path):
        print(f"Creating new output CSV: {output_csv_path}")
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
    else:
        print(f"Using existing output CSV: {output_csv_path}")
        # Verify that existing CSV has correct headers
        try:
            existing_df = pd.read_csv(output_csv_path)
            if not all(col in existing_df.columns for col in columns):
                print("Warning: Existing CSV has incorrect headers. Backing up and recreating...")
                # Backup existing file
                backup_path = output_csv_path.replace('.csv', '_backup.csv')
                os.rename(output_csv_path, backup_path)
                print(f"Backed up existing file to: {backup_path}")
                
                # Create new file with correct headers
                with open(output_csv_path, mode='w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Creating new file...")
            with open(output_csv_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=columns)
                writer.writeheader()
    
    nodule_voxel_coor = {}
    
    print(f"Extracting images and coordinates from {subset_name}")
    
    for fcount, img_file in enumerate(tqdm(file_list, desc="Processing CT files")):
        mini_df = nodule_annotations[nodule_annotations["file"] == img_file]
        
        if len(mini_df) > 0:
            try:
                # Read CT scan
                itk_img = sitk.ReadImage(img_file)
                img_array = sitk.GetArrayFromImage(itk_img)
                num_z, height, width = img_array.shape
                origin = np.array(itk_img.GetOrigin())
                spacing = np.array(itk_img.GetSpacing())
                
                for node_idx, cur_row in mini_df.iterrows():
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    diam = cur_row["diameter_mm"]
                    
                    center = np.array([node_x, node_y, node_z])
                    v_center = np.rint((center - origin) / spacing)
                    
                    # Only process the slice containing the nodule center
                    i_z = int(v_center[2])
                    if 0 <= i_z < num_z:
                        _, voxel_coor = make_mask(center, diam, i_z*spacing[2]+origin[2], 
                                         width, height, spacing, origin)
                        
                        # Generate unique ID and filename (include subset name to avoid conflicts)
                        base_name = os.path.basename(img_file).rsplit('.', 1)[0]
                        slice = f"{subset_name}_{base_name}_{i_z}_{node_idx}"
                        img_name = f"{slice}.png"
                        
                        # Normalize and save image
                        img = normalize(img_array[i_z])
                        img = img * 255
                        img_rgb = np.stack((img,)*3, axis=-1)
                        
                        img_path = os.path.join(ct_images_dir, img_name)
                        cv2.imwrite(img_path, img_rgb)
                        
                        # Store ROI coordinates
                        nodule_voxel_coor[slice] = voxel_coor
                        
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue
    
    # Save ROI coordinates to CSV
    if nodule_voxel_coor:
        df_voxel_coor = pd.DataFrame.from_dict(nodule_voxel_coor, 
                                       columns=["x_min", "x_max", "y_min", "y_max"], 
                                       orient='index')
        df_voxel_coor["ID"] = list(nodule_voxel_coor.keys())
        df_voxel_coor = df_voxel_coor.reset_index(drop=True)
        df_voxel_coor = df_voxel_coor[(df_voxel_coor['x_min'] >= 0) & (df_voxel_coor['x_max'] >= 0) & (df_voxel_coor['y_min'] >= 0) & (df_voxel_coor['y_max'] >= 0)]
        
        ids = df_voxel_coor["ID"].to_numpy()
        file_list = os.listdir(ct_images_dir)
        for file in tqdm(file_list):
            file_name = file.rsplit(".", 1)[0]
            if file_name not in ids:
                os.remove(os.path.join(ct_images_dir, file))
                print(f"Delete {file_name}")
        
        # Append to existing CSV
        df_roi = pd.read_csv(output_csv_path)
        df_roi = pd.concat([df_roi, df_voxel_coor], ignore_index=True)
        df_roi.to_csv(output_csv_path, index=False)
        
        # print(f"Processed {len(nodule_voxel_coor)} nodules from {subset_name}")
        print(f"Results saved to {current_output_dir}")
    else:
        print("No nodules were processed")

def parse_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="LUNA16 Dataset Downloader and Processor")
    
    parser.add_argument("--subset", type=str, default="subset0", 
                       help="Subset name to download (e.g., subset0, subset1, ...)")
    parser.add_argument("--data-dir", type=str, default="./data", 
                       help="Directory to store downloaded data")
    parser.add_argument("--output-dir", type=str, default="./output", 
                       help="Directory to store processed results")
    parser.add_argument("--chunk-size", type=int, default=1024*1024, 
                       help="Download chunk size in bytes")
    parser.add_argument("--download-only", action="store_true", 
                       help="Only download data without processing")
    parser.add_argument("--process-only", action="store_true", 
                       help="Only process existing data without downloading")
    parser.add_argument("--clean-output", action="store_true", 
                       help="Clean output directory before processing")
    parser.add_argument("--separate-subsets", action="store_true", 
                       help="Create separate subdirectories for each subset")
    
    return parser.parse_args()

def main():
    """Main function to orchestrate the entire pipeline"""
    opt = parse_opt()
    
    # Create directories
    os.makedirs(opt.data_dir, exist_ok=True)
    os.makedirs(opt.output_dir, exist_ok=True)
    
    # URLs for LUNA16 dataset
    subset_url = f"https://zenodo.org/records/3723295/files/{opt.subset}.zip"
    annotations_url = "https://zenodo.org/records/3723295/files/annotations.csv"
    
    # Download phase
    if not opt.download_only:
        print("=== DOWNLOAD PHASE ===")
        
        # Download subset
        subset_path = os.path.join(opt.data_dir, f"{opt.subset}.zip")
        download_file(subset_url, subset_path, opt.chunk_size)
        
        # Download annotations
        annotations_path = os.path.join(opt.data_dir, "annotations.csv")
        download_file(annotations_url, annotations_path, opt.chunk_size)
        
        print("Download phase completed!")
    
    # Processing phase
    if not opt.process_only:
        print("\n=== PROCESSING PHASE ===")
        
        # Process CT data
        process_ct_data(opt.subset, opt.data_dir, opt.output_dir, opt.clean_output, opt.separate_subsets)
        
        print("Processing phase completed!")
    
    print(f"\nPipeline finished! Check results in: {opt.output_dir}")

if __name__ == "__main__":
    main()