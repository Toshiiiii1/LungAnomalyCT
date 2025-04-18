## LUNA16 dataset

Click [here](https://luna16.grand-challenge.org/) to learn more about the LUNA16 dataset.

Lung CT scans are extracted using the SimpleITK library.

Each notebook in the `LUNA16_Image_Extraction` folder is used to extract CT scans from one of the 10 dataset folds.

The extraction result consists of two parts:
1. A folder containing all CT scan slices with nodules in `.png` format for the corresponding fold.
2. A `.csv` file containing the bounding box coordinates of all detected nodules.

The extraction process is done in Kaggle.