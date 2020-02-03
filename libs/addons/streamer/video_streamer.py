import cv2 as cv
from redis import StrictRedis
import time
from multiprocessing import Process
from libs.addons.redis.translator import frame_producer, redis_get, redis_set
from libs.settings import common_settings
from utils.utils import *
from models import *  # set ONNX_EXPORT in models.py

class VideoStreamer:
    def __init__(self, opt):
        self.opt = opt
        self.save_path = opt.output_folder
        self.__set_redis()

        self.is_running = True
        self.max_frames = opt.max_frames
        self.min_frames = opt.min_frames

        # set waiting time based on the worker type: CPU or GPU
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        # heartbeat msg to check worker status
        if self.device == "cuda":
            self.wait_time = common_settings["redis_config"]["heartbeat"]["gpu"]
        else:
            self.wait_time = common_settings["redis_config"]["heartbeat"]["cpu"]

        self.worker_id = 0 # reset, when total workers = `self.opt.total_workers`

        # Empty folders
        out_folder = opt.output_folder + str(opt.drone_id)
        if os.path.exists(out_folder):
            shutil.rmtree(out_folder)  # delete output folder
        os.makedirs(out_folder)  # make new output folder

    def __set_redis(self):
        self.rc = StrictRedis(
            host=common_settings["redis_config"]["hostname"],
            port=common_settings["redis_config"]["port"],
            password=common_settings["redis_config"]["password"],
            db=common_settings["redis_config"]["db"],
            decode_responses=True
        )

        self.rc_data = StrictRedis(
            host=common_settings["redis_config"]["hostname"],
            port=common_settings["redis_config"]["port"],
            password=common_settings["redis_config"]["password"],
            db=common_settings["redis_config"]["db_data"],
            decode_responses=True
        )

        self.rc_latency = StrictRedis(
            host=common_settings["redis_config"]["hostname"],
            port=common_settings["redis_config"]["port"],
            password=common_settings["redis_config"]["password"],
            db=common_settings["redis_config"]["db_latency"],
            decode_responses=True
        )

    def run(self):
        print("\nReading video:")
        # while True:
        while self.is_running:
            try:
                self.cap = cv.VideoCapture(self.opt.source)

                if self.opt.enable_cv_out:
                    cv.namedWindow("Image", cv.WND_PROP_FULLSCREEN)
                    cv.resizeWindow("Image", 1366, 768)  # Enter your size

                self.__start_streaming()
            except:
                print("\nUnable to communicate with the Streaming. Restarting . . .")
                # time.sleep(1) # Delay 1 second before trying again
                # The following frees up resources and closes all windows
                self.cap.release()
                if self.opt.enable_cv_out:
                    cv.destroyAllWindows()

    def __reset_worker(self):
        if self.worker_id == self.opt.total_workers:
            self.worker_id = 0

    # None = DISABLED; 1=Ready; 0=Busy
    def __worker_status(self):
        return redis_get(self.rc_data, self.worker_id)

    # Find any other available worker instead.
    def __find_optimal_worker(self):
        pass
        # finally, set avalaible worker_id into `self.worker_id`
        # when all N workers are OFF, force stop this System!

    def __load_balancing(self, frame_id, ret, frame, save_path):
        # Initially, send process into first worker
        self.worker_id += 1
        # self.worker_id = 1
        stream_channel = common_settings["redis_config"]["channel_prefix"] + str(self.worker_id)

        # Check worker status first, please wait while still working
        # print(">>>>>> __worker_status = ", self.__worker_status())
        w = 0
        wait_time = self.wait_time
        if redis_get(self.rc_data, self.worker_id) is None:
            print("This worker is OFF. Nothing to do")
            print("TBD next time: Should skip this worker and move to the next worker instead")
            self.__find_optimal_worker()
        else:
            # None = DISABLED; 1=Ready; 0=Busy
            while self.__worker_status() == 0:
                print("\nWorker-%d is still processing other image, waiting (%ds) ..." % (self.worker_id, w))
                # time.sleep(0.005)
                if not self.opt.disable_delay:
                    time.sleep(self.wait_time)
                w += self.wait_time

            # Send multi-process and set the worker as busy (value=False)
            print("### Sending the work into [worker-#%d] @ `%s`" % ((self.worker_id), stream_channel))
            Process(target=frame_producer, args=(self.rc, frame_id, ret, frame, save_path, stream_channel,
                                                 self.rc_latency, self.opt.drone_id,)).start()
            redis_set(self.rc_data, self.worker_id, 0)

        self.__reset_worker()

    def __start_streaming(self):
        n = 0
        frame_id = 0
        received_frame_id = 0
        # t_start = time.time()
        # redis_set(self.rc_latency, "start", t_start)

        # Save timestamp to start extracting video streaming.
        t_start_key = "start-" + str(self.opt.drone_id)
        redis_set(self.rc_latency, t_start_key, time.time())
        # while (self.cap.isOpened()):
        while (self.cap.isOpened()) and self.is_running:
            received_frame_id += 1

            t_sframe_key = "start-fi-" + str(self.opt.drone_id) # to calculate end2end latency each frame.
            redis_set(self.rc_latency, t_sframe_key, time.time())

            n += 1

            t0_frame = time.time()
            # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
            try:
                ret, frame = self.cap.read()

                # Latency: capture each frame
                t_frame = time.time() - t0_frame
                print('\nLatency [Reading stream image] of frame-%d: (%.5fs)' % (received_frame_id, t_frame))
                t_frame_key = "frame-" + str(self.opt.drone_id) + "-" + str(frame_id)
                redis_set(self.rc_latency, t_frame_key, t_frame)

                if n == self.opt.delay:  # read every n-th frame

                    if ret:
                        frame_id += 1

                        # Start capturing here
                        if self.min_frames == frame_id:
                            # Force stop
                            if frame_id > int(self.max_frames):
                                self.is_running = False
                                break

                            save_path = self.opt.output_folder + str(self.opt.drone_id) + "/frame-%d.jpg" % frame_id
                            self.__load_balancing(frame_id, ret, frame, save_path)

                            if self.opt.enable_cv_out:
                                cv.imshow("Image", frame)

                    else:
                        print("IMAGE is INVALID.")
                        print("I guess there is no more frame to show.")
                        break

                    n = 0
                # time.sleep(0.01)  # wait time

            except:
                print("No more frame to show.")
                break

            if cv.waitKey(10) & 0xFF == ord('q'):
                break
