from re import template
import cv2
from utils.tracker import Tracker
import numpy as np
from ex3_utils import create_gauss_peak, create_cosine_window, get_patch
from skimage import color


class MESSO(Tracker):

    def name(self):
        return 'messo'

    def __init__(self):
        self.firstInit = True

    def initialize(self, image, region):
        region = np.array(region).astype(int)

        if region[2] % 2 == 0:
            region[2] += 1
        if region[3] % 2 == 0:
            region[3] += 1

        print("New init", region)
        if False == self.firstInit:
            cv2.imshow('Example - Show image in window',self.lastR)
            cv2.waitKey(0) # waits until a key is pressed
            cv2.destroyAllWindows()
        else:
            self.firstInit = False

        self.sigma = 4
        self.lmba = 1e-3
        self.alfa = 0.1
        self.enlarge_factor = 1
        self.size = np.array([region[2], region[3]])
        self.size_enlarged = (self.size * self.enlarge_factor).astype(int)
        if self.size_enlarged[0] % 2 == 0:
            self.size_enlarged[0] += 1
        if self.size_enlarged[1] % 2 == 0:
            self.size_enlarged[1] += 1

        self.position = [region[0], region[1]]
        image, _ = get_patch(image, self.position, self.size_enlarged)

        w = self.size_enlarged[0]
        h = self.size_enlarged[1]

        image = color.rgb2gray(image)

        self.cos_window = create_cosine_window([w, h])

        image = np.multiply(image, self.cos_window)

        self.G = create_gauss_peak([w, h], self.sigma)
        #cv2.imshow('Example - Show image in window',self.G)
        #cv2.waitKey(0) # waits until a key is pressed
        #cv2.destroyAllWindows()
        self.G_fft = np.fft.fft2(self.G)
        F_fft = np.fft.fft2(image)
        F_fft_con = np.conjugate(F_fft)
        D = (np.multiply(F_fft, F_fft_con) + self.lmba)
        self.prev_H_fft_con = np.divide(np.multiply(self.G_fft, F_fft_con), D)

    def track(self, image):

        image, _ = get_patch(image, self.position, self.size_enlarged)

        w = self.size_enlarged[0]
        h = self.size_enlarged[1]

        image = color.rgb2gray(image)
        image = np.multiply(image, self.cos_window)


        F_fft = np.fft.fft2(image)

        R = np.fft.ifft2(np.multiply(self.prev_H_fft_con, F_fft)).real
        self.lastR = R

        pos_of_max = np.where(R == np.amax(R))

        maxPos = [pos_of_max[1][0], pos_of_max[0][0]]

        if maxPos[0] > w/2:
            maxPos[0] -= w

        if maxPos[1] > h/2:
            maxPos[1] -= h

        self.position[0] += maxPos[0]
        self.position[1] += maxPos[1]

        #print(self.position, maxPos)

        F_fft_con = np.conjugate(F_fft)
        D = (F_fft * F_fft_con + self.lmba)
        H_fft_con = np.divide(np.multiply(self.G_fft, F_fft_con), D)
        self.prev_H_fft_con = (1 - self.alfa) * \
            self.prev_H_fft_con + self.alfa * H_fft_con
        return [self.position[0], self.position[1], self.size[0], self.size[1]]
