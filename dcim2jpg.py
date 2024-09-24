# pip install aspose-imaging-python-net

import os
from tqdm import tqdm
from aspose.imaging import Image
from aspose.imaging.imageoptions import JpegOptions
import tkinter
from tkinter import filedialog
import cv2

dicom_extension = '.dcm'
jpeg_extension = '.jpg'

tkinter.Tk().withdraw() # Ngăn chặn cửa sổ tkinter rỗng xuất hiện
root_folder = filedialog.askdirectory() # Chọn folder chứa file DICOM

total_dicom_files = sum(
    1 for _, _, filenames in os.walk(root_folder) for filename in filenames if filename.lower().endswith(dicom_extension)
)
print('Tổng số file DICOM trong thư mục gốc: ', total_dicom_files)

def convert_DCM2JPG(INPUT, OUTPUT):      
    # Tạo và thiết lập đối tượng của lớp ImageOptionsBase, ở đây là JpegOptions
    jpeg_options = JpegOptions()
            
    # Gọi phương thức image.Save, lưu file với đường dẫn và phần mở rộng JPG
    try:
        with Image.load(INPUT) as image:
            image.save(OUTPUT, jpeg_options)
            return 1
    except RuntimeError as e:
        txt_file_path = INPUT[:-4] + '_error.txt'
        f = open(txt_file_path, 'w')
        f.write(f'{e}')
        f.close()
        return 0

def convertFolder():
    for dirpath, _, filenames in os.walk(root_folder):
        total_dicom_files_in_small_folder = sum(
            1 for filename in filenames if filename.lower().endswith(dicom_extension)
        )
        if total_dicom_files_in_small_folder == 0: continue
        
        print(f'Đang xử lý thư mục: {dirpath}')
        print(f'Tổng số file DICOM trong thư mục này: {total_dicom_files_in_small_folder}')
        for filename in tqdm(filenames, total=total_dicom_files_in_small_folder, desc='Đang thực hiện chuyển đổi'):
            if filename.lower().endswith(dicom_extension):
                dcm_file_path = os.path.join(dirpath, filename) # Load file DICOM
                jpg_file_path = dcm_file_path[:-4] + '.jpg' # Lưu file với đường dẫn và phần mở rộng JPG
                
                if convert_DCM2JPG(dcm_file_path, jpg_file_path):
                    os.remove(dcm_file_path) # Xóa file DICOM nếu chuyển đổi thành công
                    
    print('ĐÃ XONG QUÁ TRÌNH CHUYỂN ĐỔI TỪ ẢNH DICOM SANG JPEG')
    print('--------------------------------------------------')

def cropImages():
    print('BẮT ĐẦU QUÁ TRÌNH CROP ẢNH ĐỂ XÓA WATERMARK CỦA THƯ VIỆN ASPOSE.IMAGING')
    for dirpath, _, filenames in os.walk(root_folder):
        total_jpeg_files_in_small_folder = sum(
            1 for filename in filenames if filename.lower().endswith(jpeg_extension)
        )
        if total_jpeg_files_in_small_folder == 0: continue
        
        print(f'Đang xử lý thư mục: {dirpath}')
        print(f'Tổng số file JPEG trong thư mục này: {total_jpeg_files_in_small_folder}')
        for filename in tqdm(filenames, total=total_jpeg_files_in_small_folder, desc='Đang thực hiện crop ảnh'):
            if filename.lower().endswith(jpeg_extension):
                jpg_file_path = os.path.join(dirpath, filename)
                img = cv2.imread(jpg_file_path)
                if img is not None:
                    height, _, _ = img.shape
                    crop_img = img[height-460:height, :]
                    os.chdir(dirpath)
                    cv2.imwrite(jpg_file_path, crop_img)
                else:
                    print(f"Không thể đọc ảnh: {jpg_file_path}")
                
if __name__ == "__main__":
    convertFolder()
    
    cropImages()