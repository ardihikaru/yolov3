import numpy as np
from libs.algorithms.intersection_finder import IntersectionFinder

class Mbbox:
    def __init__(self, save_path, det, img, names):
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


    def run(self):
        self.__extract()
        self.intersection = IntersectionFinder(self.save_path, self.img, self.det, self.class_det, self.width, self.height)
        self.intersection.find()

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



