import enum
import numpy as np
from sklearn.preprocessing import scale
from ex2_utils import get_patch, generate_responses_1, Tracker, create_epanechnik_kernel, extract_histogram, backproject_histogram
import cv2
import os

def mean_shift_mode_seek(I, N, stop_criterion = 0.001, size_X = 5, size_Y = 5):

    number_of_rows = len(I)
    number_of_columns = len(I[0])

    ranX = range(-int(size_X/2 - 0.5), int(size_X/2 + 0.5))
    ranY = range(-int(size_Y/2 - 0.5), int(size_Y/2 + 0.5))

    Y = [[c for c in ranY] for r in ranX]
    X = [[r for c in ranY] for r in ranX]

    x_k = number_of_columns/2
    y_k = number_of_rows/2
    
    for i in range(N):
        print(i)
        w_i = get_patch(I, [round(y_k), round(x_k)], [size_X, size_Y])[0]
        x_k_new = (np.sum(np.multiply(X, w_i)))/(np.sum(w_i))
        y_k_new = (np.sum(np.multiply(Y, w_i)))/(np.sum(w_i))
        
        x_k += x_k_new
        y_k += y_k_new
        
        if np.abs(x_k_new) < stop_criterion and np.abs(y_k_new) < stop_criterion:
            return round(x_k), round(y_k)

    return round(x_k), round(y_k)


class MeanShiftTracker(Tracker):
    def __init__(self, params):
        self.eps = 1e-5
        self.nbins = 16
        self.stop_criterion = 0.5
        self.N = 20
        self.alfa = 0

    def initialize(self, image, region):

        self.originalWidth = region[2]
        self.originalHeight = region[3]

        self.scaleX = 50 / region[2]
        self.scaleY = 50 / region[3]

        newX = image.shape[1] * self.scaleX
        newY = image.shape[0] * self.scaleY

        region = [int(region[0] * self.scaleX), int(region[1] * self.scaleY), 50, 50]

        image = cv2.resize(image, dsize=(int(newX), int(newY)))


        if region[2] % 2 == 0:
            region[2] += 1
        if region[3] % 2 == 0:
            region[3] += 1


        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        left = max(region[0], 0)
        top = max(region[1], 0)

        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0] - 1)

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2], region[3])

        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], 1)

        kernel_image = image[int(top):int(bottom), int(left):int(right)]
        
        hist = extract_histogram(kernel_image, self.nbins, self.kernel)
        hist = np.divide(hist, np.sum(hist))

        self.prev_hist = hist

    def track(self, image):
        print("NEWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")

        bestSim = 999999999

        for i, scalePercent in enumerate([0.9, 1, 1.1]):
            #for j, scalePercent2 in enumerate([0.9, 1, 1.1]):
            pos, region, newHist, hist = self.trackScaled(image, [scalePercent, scalePercent])

            sim = np.sum(np.square(self.prev_hist - hist))

            if sim < bestSim:
                bestPos = pos
                bestI = i
                bestHist = newHist
                bestRegion = region
                bestSim = sim

        print(bestI)
        self.prev_hist = bestHist
        self.position = [bestPos[0], bestPos[1]]
        self.size = [bestPos[2], bestPos[3]]
        print(self.originalWidth, self.originalHeight)
        print(bestRegion)
        return bestRegion

    def trackScaled(self, image, targetScale):

        newX = int(image.shape[1] * self.scaleX)
        newY = int(image.shape[0] * self.scaleY)


        image = cv2.resize(image, dsize=(newX, newY))
        
        
        w = np.minimum(int(self.size[0] * targetScale[0]), image.shape[1] - 2)
        h = np.minimum(int(self.size[1] * targetScale[1]), image.shape[0] - 2)
        if w % 2 == 0:
            w -= 1
        if h % 2 == 0:
            h -= 1

        print(w, h)

        X, Y = np.meshgrid(np.linspace(-int(w/2), int(w/2), int(w)), np.linspace(-int(h/2), int(h/2), int(h)))

        x = int(self.position[0])
        y = int(self.position[1])

        if round(x + w/2) >= image.shape[1] - 1:
            x = int(image.shape[1] - w/2 - 1)
        if round(x - w/2) <= 0:
            x = int(w/2+1)
        if round(y + h/2) >= image.shape[0] - 1:
            print("mu")
            y = int(image.shape[0] - h/2 - 1)
        if round(y - h/2) <= 0:
            y = int(h/2+1)

        print(x, y)

        for i in range(self.N):
            print(i)


            left = max(round(x - float(w) / 2), 0)
            top = max(round(y - float(h) / 2), 0)

            right = min(left + w, image.shape[1] - 1)
            bottom = min(top + h, image.shape[0] - 1)

            print(top, left, bottom, right, top + h, image.shape[0])

            image_region = image[int(top):int(bottom), int(left):int(right)]
            
            print(image_region.shape)
            self.kernel = create_epanechnik_kernel(image_region.shape[1], image_region.shape[0], 1)
            print(image_region.shape, self.kernel.shape)
            hist = extract_histogram(image_region, self.nbins, self.kernel)
            hist = np.divide(hist, np.sum(hist))

            v = np.sqrt(np.divide(self.prev_hist, hist + self.eps))

            backprojection = backproject_histogram(image[int(top):int(bottom), int(left):int(right)], v, self.nbins)
            backprojection = np.divide(backprojection, np.sum(backprojection))

            diff_x = (np.sum(np.multiply(X, backprojection)))
            diff_y = (np.sum(np.multiply(Y, backprojection)))

            x += diff_x
            y += diff_y

            if round(x + w/2) > image.shape[1] - 1:
                x = image.shape[1] - w/2 - 1
            if round(x - w/2) <= 0:
                x = w/2+1
            if round(y + h/2) > image.shape[0] - 1:
                y = image.shape[0] - h/2 - 1
            if round(y - h/2) <= 0:
                y = h/2+1

            if np.sqrt(diff_x ** 2 + diff_y ** 2) <= self.stop_criterion:
                break

        return [x, y, w, h], [round((x - w/2) / self.scaleX), round((y - h/2)/self.scaleY), int(w/self.scaleX), int(h/self.scaleY)], (1 - self.alfa) * self.prev_hist + self.alfa * hist, hist

class MSParams():
    def __init__(self):
        self.enlarge_factor = 2

