## Preprocessed

The `preprocessed_img3` folder contains all CT scan slices with nodules from the LUNA16 dataset. The file `ROI_coor4.csv` includes the bounding box coordinates of all detected nodules.

The bounding box coordinates are formatted as: `x_min, x_max, y_min, y_max`.

The `yolo_input_prepare` script is used to convert bounding box coordinates to YOLO format. Each set of coordinates is saved in a separate `.txt` file, and all `.txt` files are stored in the `labels3` folder.

`COCO format` is used for DETR model, `FRCNN format` is used for Faster R-CNN model and `YOLO format` is used to YOLO models.