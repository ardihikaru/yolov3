import cv2
import numpy as np

def crop_image(save_path, img, xywh, idx):
    x = xywh[0]
    y = xywh[1]
    w = xywh[2]
    h = xywh[3]
    crop_img = img[y:y + h, x:x + w]
    crop_save_path = save_path.replace('.jpg', '')+"-%s-crop.jpg" % str(idx)
    cv2.imwrite(crop_save_path, crop_img)

def np_xyxy2xywh(xyxy, data_type=int):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    xywh = np.zeros_like(xyxy)
    x1 = xyxy[0]
    y1 = xyxy[1]
    x2 = xyxy[2]
    y2 = xyxy[3]

    xywh[0] = xyxy[0]
    xywh[1] = xyxy[1]
    xywh[2] = data_type(abs(x2 - x1))
    xywh[3] = data_type(abs(y1 - y2))
    return xywh
