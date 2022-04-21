import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    current_w = 0
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        keypoints1, descriptors1 = orb.detectAndCompute(im1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2, None)
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append((keypoints1[m.queryIdx].pt, keypoints2[m.trainIdx].pt))
        good = np.array(good)

        # TODO: 2. apply RANSAC to choose best H
        num_points = len(good)
        num_best_inliers = 0
        best_H = np.identity(3)
        for i in range(2000):
            points = np.array(good[random.sample(range(len(good)), 4)])
            H = solve_homography(points[:, 1], points[:, 0])
            if np.linalg.matrix_rank(H) >= 3:
                all_u = np.concatenate((good[:, 0], np.ones((num_points, 1))), axis=1)
                all_v = np.concatenate((good[:, 1], np.ones((num_points, 1))), axis=1)
                estimate_u = H @ all_v.T
                estimate_u /= estimate_u[-1]
                errors = np.linalg.norm(all_u - estimate_u.T, ord=1, axis=1)
                num_inliers = sum(errors < 4)
                if num_inliers > num_best_inliers:
                    num_best_inliers = num_inliers
                    best_H = H

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H

        # TODO: 4. apply warping
        current_w += im1.shape[1]
        out = warping(im2, dst, last_best_H, 0, im2.shape[0], current_w, current_w + im2.shape[1], direction='b') 

    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)