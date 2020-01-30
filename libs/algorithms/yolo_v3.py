from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from libs.commons.opencv_helpers import *

from libs.algorithms.mbbox import Mbbox

class YOLOv3:
    def __init__(self, opt):
        self.n_labels = 2 # This is fixed, since we desire 2 labels: Person and Flag
        self.opt = opt
        self.save_path = None
        self.t0 = None
        self.str_output = ""

        # (320, 192) or (416, 256) or (608, 352) for (height, width)
        self.img_size = (320, 192) if ONNX_EXPORT else opt.img_size

        self.out, self.source, self.weights, self.half, self.view_img, self.save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
        self.webcam = self.source == '0' or self.source.startswith('rtsp') or self.source.startswith('http') or self.source.endswith('.txt')

        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
        if os.path.exists(self.out):
            shutil.rmtree(self.out)  # delete output folder
        os.makedirs(self.out)  # make new output folder

        # Initialize model
        self.model = Darknet(opt.cfg, self.img_size)
        self.mbbox = None # Merge Bounding Box

        # Sef Default Detection Algorithms
        self.default_algorithm = opt.default_detection
        self.mbbox_algorithm = opt.mbbox_detection

    def run(self):
        print("Starting YOLO-v3 Detection Network")
        self.__load_weight()
        self.__second_stage_classifier()
        self.__eval_model()
        # self.__export_mode() # TBD
        self.__half_precision()
        self.__set_data_loader()
        self.__get_names_colors()
        self.__print_save_txt_img()
        self.__iterate_frames() # Perform detection in each frame here

        print('Done. Total elapsed time: (%.3fs)' % (time.time() - self.t0))

    def __load_weight(self):
        # Load weights
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

    def __second_stage_classifier(self):
        # Second-stage classifier
        self.classify = False
        if self.classify:
            self.modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model'])  # load weights
            self.modelc.to(self.device).eval()

    def __eval_model(self):
        # Eval mode
        self.model.to(self.device).eval()

    # Optional
    def __export_mode(self):
        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3) + self.img_size)  # (1, 3, 320, 192)
            torch.onnx.export(self.model, img, 'weights/export.onnx', verbose=False, opset_version=10)

            # Validate exported model
            import onnx
            model = onnx.load('weights/export.onnx')  # Load the ONNX model
            onnx.checker.check_model(model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
            return

    def __half_precision(self):
        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()

    def __set_data_loader(self):
        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        if self.webcam:
            self.view_img = True
            torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.img_size, half=self.half)
        else:
            self.save_img = True
            self.dataset = LoadImages(self.source, img_size=self.img_size, half=self.half)

    def __get_names_colors(self):
        # Get names and colors
        self.names = load_classes(self.opt.names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]

    # Executed after detect()
    def __print_save_txt_img(self):
        if self.save_txt or self.save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + self.out)
            if platform == 'darwin':  # MacOS
                os.system('open ' + self.out + ' ' + self.save_path)

    def __save_cropped_img(self, xyxy, im0, idx):
        if self.opt.crop_img:
            # Try saving cropped image
            original_img = im0.copy()
            numpy_xyxy = torch2numpy(xyxy, int)
            xywh = np_xyxy2xywh(numpy_xyxy)
            crop_image(self.save_path, original_img, xywh, idx)

    def __save_results(self, im0, vid_cap):
        # Save results (image with detections)
        if self.save_img:
            if self.dataset.mode == 'images':
                cv2.imwrite(self.save_path, im0)
            else:
                if self.vid_path != self.save_path:  # new video
                    self.vid_path = self.save_path
                    if isinstance(self.vid_writer, cv2.VideoWriter):
                        self.vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.vid_writer = cv2.VideoWriter(self.save_path,
                                                      cv2.VideoWriter_fourcc(*self.opt.fourcc), fps, (w, h))
                self.vid_writer.write(im0)

    def __iterate_frames(self):
        # Run inference
        self.t0 = time.time()
        for path, img, im0s, vid_cap in self.dataset:
            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(self.device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = self.model(img)[0]

            if self.opt.half:
                pred = pred.float()

            # Apply NMS: Non-Maximum Suppression
            # to Removes detections with lower object confidence score than 'conf_thres'
            self.pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                            agnostic=self.opt.agnostic_nms)

            # Apply Classifier
            if self.classify:
                self.pred = apply_classifier(pred, self.modelc, img, im0s)

            # Process detections
            '''
            p = path
            s = string for printing
            im0 = image (matrix)
            '''
            for i, det in enumerate(self.pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, self.str_output, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, self.str_output, im0 = path, '', im0s

                self.save_path = str(Path(self.out) / Path(p).name)
                self.str_output += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    if self.default_algorithm:
                        self.__default_detection(det, im0)

                    if self.mbbox_algorithm:
                        self.__mbbox_detection(det, im0) # modifying Mb-box

                    # Print time (inference + NMS)
                    print('%sDone. (%.3fs)' % (self.str_output, time.time() - t))

                    # Stream results
                    if self.view_img:
                        cv2.imshow(p, im0)
                        if cv2.waitKey(1) == ord('q'):  # q to quit
                            raise StopIteration

                    self.__save_results(im0, vid_cap)

    def __default_detection(self, det, im0):
        if self.default_algorithm:
            original_img = im0.copy()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                self.str_output += '%g %ss, ' % (n, self.names[int(c)])  # add to string

            # Write results
            idx_detected = 0
            for *xyxy, conf, cls in det:
                idx_detected += 1
                self.save_txt = True  # Ardi: manually added
                if self.save_txt:  # Write to file
                    with open(self.save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                self.__save_cropped_img(xyxy, original_img, idx_detected)

                if self.save_img or self.view_img:  # Add bbox to image
                    label = '%s %.2f' % (self.names[int(cls)], conf)
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)])

    '''
    FYI:
        Variable `det` consists of three parameters:
        1. *xyxy : Coordinate of (x1, y1) and (x2, y2)
        2. conf  : Confidence Score
        3. cls   : Class (`Person` and `Flag`)
    '''
    def __mbbox_detection(self, det, im0):
        if self.mbbox_algorithm:
            original_img = im0.copy()

            self.mbbox = Mbbox(self.save_path, det, original_img, self.names)
            self.mbbox.run()

            # extract person and flag detected objects
            # for c in det[:, -1].unique():
            #     pid_det[self.names[int(c)]] = [d for d in det if d[-1] == c]

            # print("\n pid_det = ", len(pid_det), pid_det)
            # print("\n pid_det = ", len(pid_det), pid_det)
            # print(det)
