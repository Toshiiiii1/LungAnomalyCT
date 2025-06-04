import pandas as pd
from tqdm import tqdm
import argparse
import os

def yolo_format_calculate(x_min, x_max, y_min, y_max, width, height):
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_center = x_min + (box_width/2)
    y_center = y_min + (box_height/2)

    return x_center/width, y_center/height, box_width/width, box_height/height

def parse_opt():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Convert coor format to YOLO format")
    
    parser.add_argument("--csv-file", type=str, default="output.csv", 
                       help="CSV file contain voxel coordinates")
    parser.add_argument("--output-dir", type=str, default="./output", 
                       help="Directory to store processed results")
    
    return parser.parse_args()

def main():
    opt = parse_opt()
    
    os.makedirs(opt.output_dir, exist_ok=True)
    
    df_roi = pd.read_csv(opt.csv_file)
    for idx, row in tqdm(df_roi.iterrows()):
        x_min = row["x_min"]
        x_max = row["x_max"]
        y_min = row["y_min"]
        y_max = row["y_max"]
        roi_id = row["ID"]
        
        x_center, y_center, box_w, box_h = yolo_format_calculate(x_min, x_max, y_min, y_max, 512, 512)
        
        with open(f"{opt.output_dir}/{roi_id}.txt", "w") as f:
            f.write(f"0 {x_center} {y_center} {box_w} {box_h}")

if __name__ == "__main__":
    main()
        