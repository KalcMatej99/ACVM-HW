import numpy as np
from ex2_utils import get_patch, generate_responses_1, Tracker, create_epanechnik_kernel, extract_histogram, backproject_histogram, generate_responses_2
import cv2
import os

def mean_shift_mode_seek(I, N, stop_criterion = 0.001, size_X = 5, size_Y = 5, startX = 2, startY = 2):

    ranX = range(-int(size_X/2 - 0.5), int(size_X/2 + 0.5))
    ranY = range(-int(size_Y/2 - 0.5), int(size_Y/2 + 0.5))

    Y = [[c for c in ranY] for r in ranX]
    X = [[r for c in ranY] for r in ranX]

    x_k = startX
    y_k = startY
    

    for i in range(N):
        n_iters = i + 1
        w_i = get_patch(I, [round(y_k), round(x_k)], [size_X, size_Y])[0]
        
        if np.sum(w_i) == 0:
            break
        x_k_new = (np.sum(np.multiply(X, w_i)))/(np.sum(w_i))
        y_k_new = (np.sum(np.multiply(Y, w_i)))/(np.sum(w_i))
        
        
        x_k += x_k_new
        y_k += y_k_new
        
        if np.abs(x_k_new) < stop_criterion and np.abs(y_k_new) < stop_criterion:
            break

    return [round(x_k), round(y_k)], n_iters


class MeanShiftTracker(Tracker):
    def __init__(self, params):
        self.eps = params.eps
        self.nbins = params.nbins
        self.stop_criterion = params.stop_criterion
        self.N = params.N
        self.alfa = params.alfa
        self.sigma = params.sigma

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

        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.sigma)

        kernel_image = image[int(top):int(bottom), int(left):int(right)]
        
        hist = extract_histogram(kernel_image, self.nbins, self.kernel)
        hist = np.divide(hist, np.sum(hist))

        self.prev_hist = hist

    def track(self, image):

        bestSim = 999999999
        scales = [0.90, 1, 1.1]
        scales = [1]
        for i, scalePercent in enumerate(scales):
            pos, region, newHist, hist = self.trackScaled(image, [scalePercent, scalePercent])

            sim = np.sum(np.square(self.prev_hist - hist))


            if sim < bestSim:
                bestPos = pos
                bestHist = newHist
                bestRegion = region
                bestSim = sim

        
        self.prev_hist = bestHist
        self.position = [bestPos[0], bestPos[1]]
        self.size = [bestPos[2], bestPos[3]]
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


        X, Y = np.meshgrid(np.linspace(-int(w/2), int(w/2), int(w)), np.linspace(-int(h/2), int(h/2), int(h)))

        x = int(self.position[0])
        y = int(self.position[1])

        if round(x + w/2) >= image.shape[1] - 1:
            x = int(image.shape[1] - w/2 - 1)
        if round(x - w/2) <= 0:
            x = int(w/2+1)
        if round(y + h/2) >= image.shape[0] - 1:
            y = int(image.shape[0] - h/2 - 1)
        if round(y - h/2) <= 0:
            y = int(h/2+1)


        for i in range(self.N):
            left = max(round(x - float(w) / 2), 0)
            top = max(round(y - float(h) / 2), 0)

            right = min(left + w, image.shape[1] - 1)
            bottom = min(top + h, image.shape[0] - 1)


            image_region = image[int(top):int(bottom), int(left):int(right)]
            
            self.kernel = create_epanechnik_kernel(image_region.shape[1], image_region.shape[0], self.sigma)
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
        self.eps = 1e-5
        self.nbins = 4
        self.stop_criterion = 0.01
        self.N = 20
        self.alfa = 0
        self.sigma = 0.5



if __name__ == "__main__":
    data = generate_responses_1()
    
    import cv2
    #cv2.imshow('image',data * 255.0)
    #cv2.waitKey(0)

    Ns = [1000]
    stops = [1, 0.1, 0.01]
    sizes = [5, 15, 41]
    startsX = [25, 50, 75]
    startsY = [25, 50, 75]


    for N in Ns:
        for stop in stops:
            for size in sizes:
                for startX in startsX:
                    for startY in startsY:
                        x_k, n_iters = mean_shift_mode_seek(data, N, stop, size, size, startX, startY)
                        if np.linalg.norm(np.array(x_k) - np.array([70, 50])) < 3:
                            print(stop, size, "(" + str(startX) + "," + str(startY) + ")", "(" + str(x_k[0]) + "," + str(x_k[1]) + ")", n_iters)
