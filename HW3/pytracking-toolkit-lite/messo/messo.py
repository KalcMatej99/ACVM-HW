from re import template
import cv2
from utils.tracker import Tracker
import numpy as np
from ex3_utils import create_gauss_peak, create_cosine_window


class MESSO(Tracker):

    def name(self):
        return 'messo'

    def rgb2gray(self, img):
        return [[np.average(elem_in_row) for elem_in_row in row] for row in img]

    def initialize(self, image, region):
        region = np.round(region)

        if region[2] % 2 == 0: region[2] += 1
        if region[3] % 2 == 0: region[3] += 1

        print("New init", region)

        self.sigma = 4
        self.lmba = 1
        self.alfa = 0.3
        self.enlarge_factor = 1
        self.size = [region[2], region[3]]
        self.size_enlarged = self.size * self.enlarge_factor

        left = max(region[0], 0)
        top = max(region[1], 0)

        if left == 0:
            right = self.size_enlarged[0]
        else:
            right = min(region[0] + self.size_enlarged[0], image.shape[1] - 1)

            if right == image.shape[1] - 1:
                left = right - self.size_enlarged[0]

        if top == 0:
            bottom = self.size_enlarged[1]
        else:
            bottom = min(region[1] + self.size_enlarged[1], image.shape[0] - 1)

            if bottom == image.shape[0] - 1:
                top = bottom - self.size_enlarged[1]


        image = image[int(top):int(bottom), int(left):int(right)]
        self.position = [int(region[0]), int(region[1])]

        h = int(len(image))
        w = int(len(image[0]))


        if h % 2 == 0: image = image[:h-1, :]
        if w % 2 == 0: image = image[:, :w-1]
        h = int(len(image))
        w = int(len(image[0]))

        image = self.rgb2gray(image)

        self.cos_window = create_cosine_window([w, h])

        image = np.multiply(image, self.cos_window)

        self.G = create_gauss_peak([w, h], self.sigma)
        self.G_fft = np.fft.fft2(self.G)
        F_fft = np.fft.fft2(image)
        F_fft_con = np.conjugate(F_fft)
        D = (np.multiply(F_fft, F_fft_con) + self.lmba)
        self.prev_H_fft_con = np.divide(np.multiply(self.G_fft, F_fft_con), D)

    def track(self, image):

        left = max(self.position[0], 0)
        top = max(self.position[1], 0)

        if left == 0:
            right = self.size_enlarged[0]
        else:
            right = min(left + self.size_enlarged[0], image.shape[1] - 1)

            if right == image.shape[1] - 1:
                left = right - self.size_enlarged[0]

        if top == 0:
            bottom = self.size_enlarged[1]
        else:
            bottom = min(top + self.size_enlarged[1], image.shape[0] - 1)

            if bottom == image.shape[0] - 1:
                top = bottom - self.size_enlarged[1]

        image = image[int(top):int(bottom), int(left):int(right)]

        h = int(len(image))
        w = int(len(image[0]))


        if h % 2 == 0: image = image[:h-1, :]
        if w % 2 == 0: image = image[:, :w-1]
        h = int(len(image))
        w = int(len(image[0]))

        image = self.rgb2gray(image)
        image = np.multiply(image, self.cos_window)

        
        F_fft = np.fft.fft2(image)
        F_fft_con = np.conjugate(F_fft)
        D = (np.multiply(F_fft, F_fft_con) + self.lmba)
        H_fft_con = np.divide(np.multiply(self.G_fft, F_fft_con), D)

        R = np.fft.ifft2(np.multiply(self.prev_H_fft_con, F_fft))

        pos_of_max = np.where(R == np.amax(R))

        maxPos = [pos_of_max[1][0], pos_of_max[0][0]]

        #print(maxPos, w, h)
        if maxPos[0] > w/2:
            maxPos[0] -= w

        if maxPos[1] > h/2:
            maxPos[1] -= h

        self.position[0] += maxPos[0]
        self.position[1] += maxPos[1]
        #print(self.position, maxPos)

        self.prev_H_fft_con = (1 - self.alfa) * self.prev_H_fft_con + self.alfa * H_fft_con
        return [self.position[0], self.position[1], self.size[0], self.size[1]]
