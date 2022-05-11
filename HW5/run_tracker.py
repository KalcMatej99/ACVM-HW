import argparse
import math
import os
import cv2
import numpy as np

from tools.sequence_utils import VOTSequence
from tools.sequence_utils import save_results
from siamfc import TrackerSiamFC
import random


def evaluate_tracker(dataset_path, network_path, results_dir, visualize, treshold, n_redetections, gauss_sampling):
    treshold = float(treshold)
    n_redetections = int(n_redetections)
    #gauss_sampling = int(gauss_sampling)
    sequences = []
    with open(os.path.join(dataset_path, 'list.txt'), 'r') as f:
        for line in f.readlines():
            sequences.append(line.strip())

    tracker = TrackerSiamFC(net_path=network_path)

    for sequence_name in sequences:
        
        print('Processing sequence:', sequence_name)

        bboxes_path = os.path.join(results_dir, '%s_bboxes.txt' % sequence_name)
        scores_path = os.path.join(results_dir, '%s_scores.txt' % sequence_name)

        if os.path.exists(bboxes_path) and os.path.exists(scores_path):
            print('Results on this sequence already exists. Skipping.')
            continue
        
        sequence = VOTSequence(dataset_path, sequence_name)

        img = cv2.imread(sequence.frame(0))
        gt_rect = sequence.get_annotation(0)
        tracker.init(img, gt_rect)
        results = [gt_rect]
        scores = [[10000]]  # a very large number - very confident at initialization
        #start_variance = gauss_sampling

        if visualize:
            cv2.namedWindow('win', cv2.WINDOW_AUTOSIZE)
        for i in range(1, sequence.length()):

            img = cv2.imread(sequence.frame(i))
            prediction, score = tracker.update(img, change_scale_if_score_value = treshold)

            if score < treshold:
                possible_xs = img.shape[0]
                possible_ys = img.shape[1]

                best_score = score
                best_center = tracker.center
                for i in range(n_redetections):
                    if False:
                        selected_x, selected_y = np.random.multivariate_normal(tracker.center, [[start_variance, 0], [0, start_variance]], 1)[0]

                        if selected_x < 0:
                            selected_x = 0.0

                        if selected_y < 0:
                            selected_y = 0.0
                        
                        if selected_x > possible_xs:
                            selected_x = float(possible_xs - 2)

                        if selected_y > possible_ys:
                            selected_y = float(possible_ys - 2)

                    else:
                        selected_x = float(random.sample(list(range(possible_xs)), k=1)[0])
                        selected_y = float(random.sample(list(range(possible_ys)), k=1)[0])
                    
                    tracker.center = np.array([selected_x, selected_y])
                    prediction, score = tracker.update(img, change_scale=False)
                    if score > best_score:
                        best_score = score
                        best_center = np.array([selected_x, selected_y])
                score = best_score
                if score > treshold:
                    tracker.center = best_center
                
                #start_variance += gauss_sampling

            if score > treshold:
                #start_variance = gauss_sampling
                results.append(prediction)
                scores.append([score])

                if visualize:
                    tl_ = (int(round(prediction[0])), int(round(prediction[1])))
                    br_ = (int(round(prediction[0] + prediction[2])), int(round(prediction[1] + prediction[3])))
                    cv2.rectangle(img, tl_, br_, (0, 0, 255), 1)

                    cv2.imshow('win', img)
                    key_ = cv2.waitKey(10)
                    if key_ == 27:
                        exit(0)
            else:
                results.append([math.nan, math.nan, math.nan, math.nan])
                scores.append([score])

        save_results(results, bboxes_path)
        save_results(scores, scores_path)


parser = argparse.ArgumentParser(description='SiamFC Runner Script')

parser.add_argument("--dataset", help="Path to the dataset", required=True, action='store')
parser.add_argument("--net", help="Path to the pre-trained network", required=True, action='store')
parser.add_argument("--results_dir", help="Path to the directory to store the results", required=True, action='store')
parser.add_argument("--visualize", help="Show ground-truth annotations", required=False, action='store_true')
parser.add_argument("--treshold", help="Treshold of the long term tracker", required=True, action='store')
parser.add_argument("--n_redetections", help="Number of redetection of long term tracker", required=True, action='store')
parser.add_argument("--gauss_sampling", help="Use gaussian sampling in the long term tracker", required=False, action='store')

args = parser.parse_args()

evaluate_tracker(args.dataset, args.net, args.results_dir, args.visualize, args.treshold, args.n_redetections, args.gauss_sampling)
