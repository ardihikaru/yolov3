from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2 as cv
import simplejson as json
from redis import StrictRedis

from multiprocessing import Process

# def frame_producer(my_redis, ret, frame):
def frame_producer(my_redis, ret, frame):
    if ret:
        # Save image
        t0 = time.time()
        save_path = "/home/ardi/devel/nctu/5g-dive/docker-yolov3/output_frames/hasil.jpg"
        cv2.imwrite(save_path, frame)
        print(".. image is saved in (%.3fs)" % (time.time() - t0))

        # Publish information
        t0 = time.time()
        data = {
            "frame_id": 1,
            "img_path": save_path
        }
        p_mdata = json.dumps(data)
        print(" .. Start publishing")
        my_redis.publish('stream', p_mdata)
        print(".. frame is published in (%.3fs)" % (time.time() - t0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--delay", type=int, default=4, help="path to dataset")
    parser.add_argument("--delay", type=int, default=7, help="path to dataset")
    parser.add_argument("--output_folder", type=str, default="data/samples", help="path to dataset")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--source", type=str, default="data/5g-dive/customTest_MIRC-Roadside-5s.mp4", help="source")
    # parser.add_argument("--source", type=str, default="http://140.113.86.92:10000/drone-3.flv", help="source")
    # parser.add_argument("--source", type=str, default="http://140.113.86.92:10000/drone-2.flv", help="source")
    parser.add_argument("--source", type=str, default="http://140.113.86.92:10000/drone-1.flv", help="source")
    opt = parser.parse_args()
    print(opt)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # os.makedirs("output", exist_ok=True)

    # dataloader = DataLoader(
    #     ImageFolder(opt.image_folder, img_size=opt.img_size),
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.n_cpu,
    # )

    rc = StrictRedis(
        host="localhost",
        port=6379,
        password="bismillah",
        db=0,
        decode_responses=True
    )

    # Ardi: Use video instead
    print("\nReading video:")
    cap = cv.VideoCapture(opt.source)
    cv.namedWindow("Image", cv.WND_PROP_FULLSCREEN)
    cv.resizeWindow("Image", 1366, 768) # Enter your size
    prev_time = time.time()
    # frame_id = 0

    sekali = True
    n = 0
    while (cap.isOpened()):
        # frame_id += 1
        n += 1

        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        try:
            ret, frame = cap.read()

            if n == opt.delay:  # read every n-th frame

                if sekali:
                    Process(target=frame_producer, args=(rc, ret, frame,)).start()
                    sekali = False

                # save_path = "/home/ardi/devel/nctu/5g-dive/docker-yolov3/output_frames/"
                # frame_save_path = save_path + "frame-%d.jpg" % frame_id
                # cv.imwrite(frame_save_path, frame)
                if ret:
                    cv.imshow("Image", frame)

                n = 0

            time.sleep(0.01)  # wait time

        except:
            print("No more frame to show.")
            break

        if cv.waitKey(10) & 0xFF == ord('q'):
            break
    # The following frees up resources and closes all windows
    cap.release()
    cv.destroyAllWindows()
