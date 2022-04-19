import time
import cv2
from sequence_utils import VOTSequence
from particle import ParticleFilter

def run(display = True, sequence = 'bolt'):

    print("Running sequence:", sequence)
    
    dataset_path = 'data/vot2014'

    # visualization and setup parameters
    win_name = 'Tracking window'
    reinitialize = True
    show_gt = True
    video_delay = 15
    font = cv2.FONT_HERSHEY_PLAIN

    # create sequence object
    sequence = VOTSequence(dataset_path, sequence)
    init_frame = 0
    n_failures = 0
    
    tracker = ParticleFilter()

    time_all = 0

    # initialize visualization window
    if display:
        sequence.initialize_window(win_name)
    # tracking loop - goes over all frames in the video sequence
    frame_idx = 0
    while frame_idx < sequence.length():
        img = cv2.imread(sequence.frame(frame_idx))
        # initialize or track
        if frame_idx == init_frame:
            # initialize tracker (at the beginning of the sequence or after tracking failure)
            t_ = time.time()
            particles = tracker.initialize(img, sequence.get_annotation(frame_idx, type='rectangle'))
            time_all += time.time() - t_
            predicted_bbox = sequence.get_annotation(frame_idx, type='rectangle')
        else:
            # track on current frame - predict bounding box
            t_ = time.time()
            predicted_bbox, particles = tracker.track(img)
            time_all += time.time() - t_
            
        try:
            # calculate overlap (needed to determine failure of a tracker)
            gt_bb = sequence.get_annotation(frame_idx, type='rectangle')
            o = sequence.overlap(predicted_bbox, gt_bb)
        except:
            break

        # draw ground-truth and predicted bounding boxes, frame numbers and show image
        if show_gt:
            sequence.draw_region(img, gt_bb, (0, 255, 0), 1)
            
        if display:
            sequence.draw_region(img, predicted_bbox, (0, 0, 255), 2)
            sequence.draw_text(img, '%d/%d' % (frame_idx + 1, sequence.length()), (25, 25))
            sequence.draw_text(img, 'Fails: %d' % n_failures, (25, 55))

            for particle in particles:
                sequence.draw_region(img, [particle[0], particle[1], 1,1], (0, 0, 0), 2)
                #sequence.draw_region(img, [particle[0]-gt_bb[2]/2, particle[1]-gt_bb[3]/2, gt_bb[2],gt_bb[3]], (0, 0, 0), 2)

            sequence.show_image(img, video_delay)

        if o > 0 or not reinitialize:
            # increase frame counter by 1
            frame_idx += 1
        else:
            # increase frame counter by 5 and set re-initialization to the next frame
            frame_idx += 5
            init_frame = frame_idx
            n_failures += 1

    print('Tracking speed: %.1f FPS' % (sequence.length() / time_all))
    print('Tracker failed %d times' % n_failures)

    return sequence.length() / time_all, n_failures

run(True, sequence="ball")
run(False, sequence="bolt")
run(False, sequence="car")
run(False, sequence="fernando")
run(False, sequence="torus")