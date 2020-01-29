import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

from libs.algorithms.yolo_v3 import YOLOv3

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--crop_img', type=str, default=True, help='*.cfg path')
    parser.add_argument('--crop_img', type=str, default=False, help='*.cfg path')
    # parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--cfg', type=str, default='yolo-obj-v5.cfg', help='*.cfg path')
    # parser.add_argument('--names', type=str, default='data/coco.names', help='*.names path')
    parser.add_argument('--names', type=str, default='data/obj.names', help='*.names path')
    # parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--weights', type=str, default='weights/yolo-dset6-best.weights', help='path to weights file')
    # parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='data/5g-dive/57-frames', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='data/5g-dive/sample-n-frames', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='data/5g-dive/sample-4-frames', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='data/5g-dive/sample-1-frame', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='data/5g-dive/videos/customTest_MIRC-Roadside-5s.mp4', help='source')  # input file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='http://140.113.86.92:10000/drone-1.flv', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    # parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--img-size', type=int, default=832, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        YOLOv3(opt).run()
        # detect()
