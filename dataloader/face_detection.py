import cv2
from skimage import transform, feature
import numpy as np
import joblib
import matplotlib.pyplot as plt
import argparse


def sample_video(file_path, sampling_rate=20):
    """ sample frames from viedeo using given sampling rate """
    # Opens the Video file
    videofile = cv2.VideoCapture(file_path)
    # Initializes frame_sequences & index for while loop
    i = 0
    frame_seq = []
    # Looping over all the frames in the videos
    while videofile.isOpened():
        # Read a single frame
        # {ret, frame} are 2 returned values for method: videofile.read().
        # ret is a boolean value. If the given frame is valid -> True, otherwise -> False
        ret, frame = videofile.read()
        if not ret:
            # break when the video comes to the end
            break
        if i % sampling_rate == 0:  # Save one frame from every 20 frames of the video
            # change the given RGB frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # append the selected grayscale frame to the frame sequences
            frame_seq.append(gray)
        i += 1
    # When you call videofile.release(), then:
    # 1. release software resource
    # 2. release hardware resource
    # This action is to be prepared for the next VideoCapture, or else an OpenCv Exception will be raised. 
    videofile.release()
    # destroys all the windows created
    cv2.destroyAllWindows()
    return frame_seq


def sliding_window(img, patch_size, istep=2, jstep=2, scale=2.0):
    for (k, resized) in enumerate(transform.pyramid_gaussian(img, downscale=scale)):
        # if the image is smaller than filter size, break from the loop
        if resized.shape[0] < patch_size[0] or resized.shape[1] < patch_size[1]:
            break
        Ni, Nj = (int(s) for s in patch_size)

        for i in range(0, resized.shape[0] - Ni, istep):
            for j in range(0, resized.shape[1] - Ni, jstep):
                patch = resized[i:i + Ni, j:j + Nj]
                yield (i, j), patch, scale ** k


# Malisiewicz et al.
def non_max_suppression_fast(boxes, confidence, overlapThresh=0.4, reversed=False):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by confidence score
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    if reversed:
        idxs = np.argsort(-confidence.flatten())
    else:
        idxs = np.argsort(confidence.flatten())

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        if reversed:
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap == 1.0)[0])))
        else:
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), confidence[pick]


def process_frame(frame_seq, video_name, image_size, result_dir, model_path, num_person=1):
    count = 0
    model = joblib.load(model_path)
    results = np.zeros((len(frame_seq), image_size[0], image_size[1], 3), dtype=np.uint8)
    for one_frame in frame_seq:
        frame_blurred = cv2.blur(one_frame, (6, 6))
        indices, patches, scale_levels = zip(*sliding_window(frame_blurred, patch_size=image_size))
        patches_hog = np.array([feature.hog(patch) for patch in patches])

        labels = model.predict(patches_hog)
        candidate_boxes_idx = np.argwhere(labels > 0)
        candidate_boxes = []
        confidence_score = []

        indices = np.array(indices)
        scale_levels = np.array(scale_levels)

        for k in candidate_boxes_idx:
            confidence = model.decision_function(patches_hog[k])
            if confidence > 2:
                scale = scale_levels[k]
                x = indices[k][0][0] * scale
                y = indices[k][0][1] * scale
                corners = (x, y, x + image_size[0] * scale - 1, y + image_size[1] * scale - 1)
                candidate_boxes.append(corners)
                confidence_score.append(confidence)

        candidate_boxes = np.array(candidate_boxes)
        confidence_score = np.array(confidence_score)

        bounding_boxes, cs = non_max_suppression_fast(candidate_boxes, confidence_score, 0.4)
        if bounding_boxes.shape[0] > 1:
            bounding_boxes, cs = non_max_suppression_fast(bounding_boxes, cs, reversed=True)

        idxs = np.argsort(-cs.flatten())
        num_of_boxes = min(num_person, len(bounding_boxes))

        intra_frame_count = 0
        for i in range(num_of_boxes):
            coord = bounding_boxes[idxs[i]].flatten()
            crop_image = one_frame[coord[0]:coord[2], coord[1]:coord[3]]
            resized = cv2.resize(crop_image, image_size, interpolation=cv2.INTER_AREA)
            plt.imshow(resized, cmap="gray")
            plt.show()
            print(resized.shape)
            # save_file_name = video_name.split(".")[0] + "_" + str(count) + "_" + str(intra_frame_count) + ".jpg"
            # cv2.imwrite(result_dir + "/" + save_file_name, resized)
            intra_frame_count += 1
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
        results[count] = resized
        count += 1
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CSC420 Project')
    parser.add_argument('--filename', nargs=1,
                        help='Please enter video file name to process. This video should sit inside data folder.')
    parser.add_argument('--sample_rate', type=int, nargs=1,
                       help='Please specify a sampling rate.')
    args = parser.parse_args()

    if not args.filename:
        file_name = 'sheldon.mp4'
        print('No video filename specified. Using default video sheldon.mp4.')
    else:
        file_name = args.filename[0]

    if not args.sample_rate:
        sample_rate = 10
        print('No sample rate specified. Using default sample rate 10.')
    else:
        sample_rate = args.sample_rate[0]

    dir_name = '../data/'
    result_dir = '../data/detection_results'
    image_size = (48, 48)

    frames = sample_video(f"{dir_name}/{file_name}", sample_rate)
    # frames = sample_video(f"{dir_name}/{file_name}", 50)

    count = 0
    for frame in frames:
        # Display the resulting frame
        plt.imshow(frame, cmap="gray")
        plt.show()
        count += 1

    process_frame(frames, result_dir=result_dir, image_size=image_size, video_name=file_name)
