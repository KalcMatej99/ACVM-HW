from re import template
import cv2
from utils.tracker import Tracker
import numpy as np
from ex3_utils import create_gauss_peak, create_cosine_window, get_patch


class MESSO(Tracker):

    def name(self):
        return 'messo'

    def __init__(self, params = None):
        self.firstInit = True
        if params == None:
            self.sigma = 2.0
            self.lmba = 0.001
            self.alfa = 0.15
            self.enlarge_factor = 1.2
        else:
            self.sigma = params.sigma
            self.lmba = params.lmba
            self.alfa = params.alfa
            self.enlarge_factor = params.enlarge_factor
        print("PARAMS", self.sigma, self.lmba, self.alfa, self.enlarge_factor)

    def initialize(self, image, region):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        region = np.array(region).astype(int)

        self.size = np.array([region[2], region[3]])
        self.size_enlarged = (self.size * self.enlarge_factor).astype(int)

        if self.size_enlarged[0] % 2 == 0: self.size_enlarged[0] -= 1
        if self.size_enlarged[1] % 2 == 0: self.size_enlarged[1] -= 1

        self.position = [int(region[0] + region[2] / 2), int(region[1] + region[3] / 2)]
        image, _ = get_patch(image, self.position, self.size_enlarged)

        w = self.size_enlarged[0]
        h = self.size_enlarged[1]


        self.cos_window = create_cosine_window([w, h])

        image = np.multiply(image, self.cos_window)

        self.G_fft = np.fft.fft2(create_gauss_peak([w, h], self.sigma))
        F_fft = np.fft.fft2(image)
        F_fft_con = np.conjugate(F_fft)
        D = (np.multiply(F_fft, F_fft_con) + self.lmba)
        self.prev_H_fft_con = np.divide(np.multiply(self.G_fft, F_fft_con), D)

    def track(self, realImage):

        realImage = cv2.cvtColor(realImage, cv2.COLOR_BGR2GRAY)
        image, _ = get_patch(realImage, self.position, self.size_enlarged)

        w = self.size_enlarged[0]
        h = self.size_enlarged[1]

        image = np.multiply(image, self.cos_window)

        F_fft = np.fft.fft2(image)

        R = np.fft.ifft2(np.multiply(self.prev_H_fft_con, F_fft)).real

        y_shift, x_shift = np.array(np.unravel_index(R.argmax(), R.shape))

        if x_shift > w/2: x_shift -= w
        if y_shift > h/2: y_shift -= h

        self.position[0] += x_shift
        self.position[1] += y_shift

        newPatch,_ = get_patch(realImage, self.position, self.size_enlarged)
        F_fft = np.fft.fft2(np.multiply(newPatch, self.cos_window))
        F_fft_con = np.conjugate(F_fft)
        D = (F_fft * F_fft_con + self.lmba)
        H_fft_con = np.divide(np.multiply(self.G_fft, F_fft_con), D)
        self.prev_H_fft_con = (1 - self.alfa) * \
            self.prev_H_fft_con + self.alfa * H_fft_con

        return [self.position[0]-int(self.size_enlarged[0]/2), self.position[1]-int(self.size_enlarged[1]/2), self.size[0], self.size[1]]
