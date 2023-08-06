import cv2
import numpy as np
import os

def compute_precision_recall(ground_truth_folder, mask_predict_folder):
    tp = 0
    fn = 0
    fp = 0
    precisions = []
    recalls = []

    for filename in os.listdir(ground_truth_folder):
        ground_truth_path = os.path.join(ground_truth_folder, filename)
        mask_predict_path = os.path.join(mask_predict_folder, filename)

        ground_truth_img = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
        mask_predict_img = cv2.imread(mask_predict_path, cv2.IMREAD_GRAYSCALE)

        # Chuyển đổi ảnh sang dạng nhị phân
        _, ground_truth_img = cv2.threshold(ground_truth_img, 1, 255, cv2.THRESH_BINARY)
        _, mask_predict_img = cv2.threshold(mask_predict_img, 1, 255, cv2.THRESH_BINARY)

        # Tính toán giá trị TP, FN và FP
        tp += np.sum(np.logical_and(ground_truth_img == 255, mask_predict_img == 255))  # Vị trí ground_true và mask đều có pixel
        fn += np.sum(np.logical_and(ground_truth_img == 255, mask_predict_img == 0))    # Vị trí mà ground_true có mà mask không có - Không phát hiện được pixel
        fp += np.sum(np.logical_and(ground_truth_img == 0, mask_predict_img == 255))    # Vị trí mà mask có mà ground_tru k có - Phát hiện pixel k có thật

        # Tính toán Precision và Recall cho từng ảnh
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        precisions.append(precision)
        recalls.append(recall)

    return precisions, recalls

def compute_ap(precision, recall):
    m_recall = np.concatenate(([0.], recall, [1.]))
    m_precision = np.concatenate(([0.], precision, [0.]))

    for i in range(m_precision.size - 1, 0, -1):
        m_precision[i - 1] = max(m_precision[i - 1], m_precision[i])

    indices = np.where(m_recall[1:] != m_recall[:-1])[0] + 1
    ap = np.sum((m_recall[indices] - m_recall[indices - 1]) * m_precision[indices])
    return ap



# Thư mục chứa ảnh ground truth
ground_truth_folder = 'PollenSegmentation/ground_true'
# Thư mục chứa ảnh mask predict
mask_predict_folder = 'PollenSegmentation/pred'

precisions, recalls = compute_precision_recall(ground_truth_folder, mask_predict_folder)

AP = compute_ap(precisions, recalls)

print("AP: ")
print(AP)

# Chuyển đổi các mảng precision và recall về cùng một số chiều
precisions = np.expand_dims(precisions, axis=1)
recalls = np.expand_dims(recalls, axis=1)

# Tính số lượng ảnh
num_images = len(precisions)

# Tính tổng Precision và tổng Recall của từng ảnh
total_precision = np.sum(precisions, axis=0)
total_recall = np.sum(recalls, axis=0)

mean_precision = total_precision / num_images
mean_recall = total_recall / num_images

# Tính toán mAP từ Precision và Recall đã tính được


print("Mean Precision:")
print(mean_precision)

print("Mean Recall:")
print(mean_recall)

