import numpy as np
from libs.algorithms.intersection_finder import IntersectionFinder
from libs.commons.opencv_helpers import save_txt

class Mbbox:
    def __init__(self, opt, save_path, det, img, names, w_ratio, h_ratio):
        self.opt = opt
        self.img = img
        self.save_path = save_path
        self.height, self.width, self.channels = img.shape
        self.det = det # Original detected objects
        # self.pid_det = {} # PID Object = Person in Distress = Person with flag
        # self.pid_det = {}
        self.class_det = {}
        # self.persons = []
        # self.flags = []
        # self.flags = np.empty(len(det))
        self.names = names
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio


    def run(self):
        self.__extract()
        # Bug fixing: When unable to find both Person and Flag object, ignore
        if len(self.class_det) == 2:
            self.intersection = IntersectionFinder(self.opt, self.names, self.save_path, self.img, self.det, self.class_det, self.width, self.height, self.w_ratio, self.h_ratio)
            self.intersection.find()
            detected_mbbox = self.intersection.get_mbbox_imgs()
            print("Person-W-Flag object: %d founds." % len(detected_mbbox))
        else:
            save_txt(self.save_path, self.opt.txt_format)
            print("Unable to find BOTH Person and Flag object in this Frame.")

    '''
    FYI: Class label in this case (check in file `data/obj.names`):
        - class `Person` = 0; 
        - class `Flag` = 1; 
    '''
    def __extract(self):
        # for d in self.det:
        for c in self.det[:, -1].unique():
            # self.pid_det[self.names[int(c)]] = [d for d in self.det if d[-1] == c]
            # self.pid_det[self.names[int(c)]] = [i for i in range (len(self.det)) if self.det[i][-1] == c]
            self.class_det[self.names[int(c)]] = [i for i in range (len(self.det)) if self.det[i][-1] == c]
        # print("\n ## Class_Det = ", self.class_det)



