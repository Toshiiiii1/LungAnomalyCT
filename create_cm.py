import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings

OKS_SIGMA = (
    np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89])
    / 10.0
)

def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)

class ConfusionMatrix:
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45, task="detect"):
        """Initialize attributes for the YOLO model."""
        self.task = task
        self.matrix = np.zeros((nc + 1, nc + 1)) if self.task == "detect" else np.zeros((nc, nc))
        self.nc = nc  # number of classes
        self.conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
        self.iou_thres = iou_thres

    def process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (Array[M, 4]): Ground truth bounding boxes in the format [x_min, y_min, width, height].
            gt_cls (Array[M]): The class labels.
        """
        # Chuyển đổi gt_bboxes sang định dạng (x1, y1, x2, y2) để dễ tính toán IoU
        gt_bboxes_xyxy = torch.from_numpy(np.empty((gt_bboxes.shape[0], 4)))
        gt_bboxes_xyxy[:, 0] = gt_bboxes[:, 0]  # x_min
        gt_bboxes_xyxy[:, 1] = gt_bboxes[:, 1]  # y_min
        gt_bboxes_xyxy[:, 2] = gt_bboxes[:, 0] + gt_bboxes[:, 2]  # x_max = x_min + width
        gt_bboxes_xyxy[:, 3] = gt_bboxes[:, 1] + gt_bboxes[:, 3]  # y_max = y_min + height

        # Nếu không có GT
        if gt_cls.shape[0] == 0:
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for dc in detection_classes:
                    self.matrix[dc, self.nc] += 1  # false positives
            return
        
        if detections is None:
            gt_classes = gt_cls.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()

        # Tính IoU giữa gt_bboxes và detections
        iou = box_iou(gt_bboxes_xyxy, detections[:, :4])  # Sử dụng hàm box_iou tương ứng

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background
                    
    def matrix(self):
        """Returns the confusion matrix."""
        return self.matrix

    def tp_fp(self):
        """Returns true positives and false positives."""
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return (tp[:-1], fp[:-1]) if self.task == "detect" else (tp, fp)  # remove background class if task=detect

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="./", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn  # scope for faster 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    def print(self):
        """Print the confusion matrix to the console."""
        for i in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[i])))
            
if __name__ == "__main__":
    # Giả lập dữ liệu
    # Nhãn thực tế (ground truth) với định dạng gt_bboxes [x_min, y_min, width, height]
    gt_bboxes = torch.tensor([[134, 728, 36, 36]])  # Nhãn duy nhất cho ví dụ
    gt_cls = torch.tensor([0])  # Nhãn lớp tương ứng

    # Dự đoán từ mô hình với định dạng detections
    # scores, labels, boxes được định dạng là tensor
    # detections = {
    #     "scores": torch.tensor([0.9546]),
    #     "labels": torch.tensor([1]),
    #     "boxes": torch.tensor([[95.8941, 129.5206, 150.4863, 184.0439]])  # [x1, y1, x2, y2]
    # }
    
    detections = {
        "boxes": torch.tensor([[ 99.5391, 132.0737, 149.9568, 181.9751],
        [400.2238, 283.0023, 419.3425, 303.0401],
        [352.6500, 323.3779, 372.4798, 343.1810],
        [ 47.6082, 232.5733,  77.4813, 267.3208],
        [155.2806, 301.7628, 177.3529, 324.7935],
        [118.2091, 276.0301, 137.3800, 294.9810]]),
        "labels": torch.tensor([1, 1, 1, 1, 1, 1]),
        "scores": torch.tensor([0.8439, 0.3177, 0.2597, 0.0836, 0.0716, 0.0691])
    }

    # Số lượng lớp (nc)
    num_classes = 1  # Chúng ta có 2 lớp: lungct và nodule

    # Khởi tạo ma trận nhầm lẫn
    confusion_matrix = ConfusionMatrix(nc=num_classes, conf=0.5, iou_thres=0.5)

    # Chuyển đổi dự đoán thành định dạng mà lớp ConfusionMatrix yêu cầu
    pred_boxes = detections['boxes']
    pred_scores = detections['scores']
    pred_labels = detections['labels']

    # Tạo mảng dự đoán hoàn chỉnh với định dạng [x1, y1, x2, y2, conf, class]
    predictions = torch.cat([pred_boxes, pred_scores.unsqueeze(1), pred_labels.unsqueeze(1).float()], dim=1)

    # Cập nhật ma trận nhầm lẫn
    for prediction in predictions:
        print(prediction)
        # print(torch.tensor(prediction)[:, 4])

    #     confusion_matrix.process_batch(prediction, gt_bboxes, gt_cls)

    # # In ra ma trận nhầm lẫn
    # print("Confusion Matrix:")
    # print(confusion_matrix.matrix)

    # # In ra true positives và false positives
    # tp, fp = confusion_matrix.tp_fp()
    # print("True Positives:", tp)
    # print("False Positives:", fp)
    
    # confusion_matrix.plot(normalize=False)