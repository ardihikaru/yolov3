from __future__ import division
import argparse
from libs.addons.streamer.video_streamer import VideoStreamer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_delay', type=bool, default=True, help='Max Frames')
    parser.add_argument('--min_frames', type=int, default=201, help='Min Frames')
    parser.add_argument('--max_frames', type=int, default=300, help='Max Frames')
    # parser.add_argument('--max_frames', type=int, default=5, help='Max Frames')
    parser.add_argument('--drone_id', type=int, default=1, help='Drone ID')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    # parser.add_argument("--total_workers", type=int, default=3, help="path to dataset")
    # parser.add_argument("--total_workers", type=int, default=2, help="path to dataset")
    parser.add_argument("--total_workers", type=int, default=1, help="path to dataset")
    parser.add_argument("--enable_cv_out", type=bool, default=False, help="path to dataset")
    # parser.add_argument("--enable_cv_out", type=bool, default=True, help="path to dataset")
    # parser.add_argument("--delay", type=int, default=4, help="path to dataset")
    # parser.add_argument("--delay", type=int, default=7, help="path to dataset")
    parser.add_argument("--delay", type=int, default=1, help="path to dataset")
    # parser.add_argument("--output_folder", type=str, default="/home/ardi/devel/nctu/5g-dive/docker-yolov3/output_frames/", help="path to save raw images")
    parser.add_argument("--output_folder", type=str, default="output_frames/", help="path to save raw images")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--source", type=str, default="data/5g-dive/customTest_MIRC-Roadside-5s.mp4", help="source")
    # parser.add_argument("--source", type=str, default="http://140.113.86.92:10000/drone-3.flv", help="source")
    # parser.add_argument("--source", type=str, default="http://140.113.86.92:10000/drone-2.flv", help="source")
    # parser.add_argument("--source", type=str, default="http://140.113.86.92:10000/drone-1.flv", help="source")
    parser.add_argument("--source", type=str, default="http://192.168.0.50:10000/drone-1.flv", help="source")
    opt = parser.parse_args()
    print(opt)

    VideoStreamer(opt).run()
