from libs.commons.opencv_helpers import get_det_xyxy, np_xyxy2xywh, get_mbbox
from utils.utils import plot_one_box
import cv2

class IntersectionFinder:
    def __init__(self, save_path, img, det, class_det, ori_width, ori_height, w_ratio, h_ratio, version=1):
        self.img = img
        self.img_plot = img.copy()
        self.det = det
        self.class_det = class_det
        self.version = version
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio
        self.ori_width = ori_width
        self.ori_height = ori_height
        self.rgb = {
            "EnlargedPerson": [13, 67, 174],
            "Person": [198, 20, 174],
            "Flag": [167, 172, 9],
            "MMBox": [198, 50, 13]
        }
        self.save_path = save_path
        self.mbbox_imgs = []
        self.intersection_num = 0
        self.intersection_idx = []

    def find(self):
        if self.version == 1:
            self.__find_intersections_v1()

        elif self.version == 2:
            self.__find_intersections_v2()

        elif self.version == 3:
            self.__find_intersections_v3()

    def __enlarge_bbox(self, bbox):
        xywh = np_xyxy2xywh(bbox)

        added_w = int(xywh[2] * self.w_ratio)
        added_h = int(xywh[3] * self.h_ratio)

        bbox[0] = bbox[0] - added_w
        bbox[1] = bbox[1] - added_h
        bbox[2] = bbox[2] + added_w
        bbox[3] = bbox[3] + added_h

        return bbox

    '''
     (x0,y0)      
        +------------+
        |            |
        |    (x4,y4) |
        |       +----------+
        |       |    |     |
        +-------|----+     |
                |  (x2,y2) |
                |          |
                +----------+
                        (x5,y5)
    '''
    def __is_intersect(self, obj_1, obj_2):
        # obj_1: (x0, y0) and (x2, y2)
        # obj_2: (x4, y4) and (x5, y5)
        x0, y0, x2, y2 = obj_1[0], obj_1[1], obj_1[2], obj_1[3]
        x4, y4, x5, y5 = obj_2[0], obj_2[1], obj_2[2], obj_2[3]
        if max(x4, x0) > min(x2, x5) or max(y4, y0) > min(y2, y5):
            return False
        else:
            return True

    def __verify_intersection(self, flag_idx, detected_intersection):
        print("\n Number of intersection found = ", len(detected_intersection))
        # 2. Then, determine the action based on the collected intersection
        # 2.1 Found one: directly marked as MB-Box
        if len(detected_intersection) == 1:
            # 2.1.1 generate MB-Box
            this_person_idx = detected_intersection[0]
            flag_xyxy = get_det_xyxy(self.det[flag_idx])
            person_xyxy = get_det_xyxy(self.det[this_person_idx])
            mbbox_xyxy = get_mbbox(flag_xyxy, person_xyxy)

            # 2.1.2 delete this person index
            del self.class_det["Person"][this_person_idx]
            # 2.1.3 plot MB-Box
            # plot_one_box(mbbox_xyxy, self.img_plot, label="MB-Box-F%s" % str(flag_idx), color=self.rgb["MMBox"])
            plot_one_box(mbbox_xyxy, self.img_plot, label="Person-W-Flag", color=self.rgb["MMBox"])

        # 2.2 Found multi-intersection: perform kNN to get the the nearest Person object
        elif len(detected_intersection) > 1:
            pass
    '''
    1. Each PERSON: W and H enlarged based on `w_ratio` and `h_ratio`
    2. For each FLAG, Find the intersection between person and flag
    '''
    def __find_intersections_v1(self):
        for flag_idx in self.class_det["Flag"]:

            # 1. Collect total number of intersection with detected Person object
            detected_intersection = []
            for person_idx in self.class_det["Person"]:
                flag_xyxy = get_det_xyxy(self.det[flag_idx])
                person_xyxy = get_det_xyxy(self.det[person_idx])
                # plot_one_box(person_xyxy, self.img_plot, label="Person-%s" % str(person_idx), color=self.rgb["MMBox"])

                person_xyxy = self.__enlarge_bbox(person_xyxy) # enlarge bbox size

                if self.__is_intersect(flag_xyxy, person_xyxy):
                    print("\n ### YAY INTERSECTION OCCURS!")
                    detected_intersection.append(person_idx)
                    # 1. Append this `person_idx` into `self.intersection_idx`
                    # self.intersection_idx.append(person_idx)
                    # # 2. Delete this `person_idx` from `self.class_det["Person"]`
                    # print("\n >>> class person awal = ", self.class_det["Person"])
                    # del self.class_det["Person"][person_idx]
                    # print("\n >>> class person AFTER = ", self.class_det["Person"])
                    # # 3. Execute `self.intersection_num` ++
                    # # self.intersection_num += 1
                else:
                    print("\n #### NOPE.")

                # Testing only: Try plotting bounding boxes
                # plot_one_box(person_xyxy, self.img_plot, label="EnPer-%s" % str(person_idx), color=self.rgb["EnlargedPerson"])
                # plot_one_box(flag_xyxy, self.img_plot, label="Flag-%s" % str(flag_idx), color=self.rgb["Person"])

            self.__verify_intersection(flag_idx, detected_intersection)

        # save MB-Box illustration
        cv2.imwrite(self.save_path.replace('.png', '')+"-mbbox.png", self.img_plot)

    '''
    1. Each FLAG: W and H enlarged based on `w_ratio` and `h_ratio`
    2. For each FLAG, Find the intersection between person and flag
    '''
    def __find_intersections_v2(self):
        pass

    '''
    1. Each FLAG: W and H enlarged based on `h_ratio`
    2. For each FLAG, Find the intersection between person and flag
    '''
    def __find_intersections_v3(self):
        pass
