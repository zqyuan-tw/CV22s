import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = []
        for i in range(self.num_octaves):
            if i:
                image = cv2.resize(gaussian_images[i - 1][-1], (gaussian_images[i - 1][-1].shape[1] // 2, gaussian_images[i - 1][-1].shape[0] // 2), interpolation = cv2.INTER_NEAREST)
            octave = []
            for j in range(self.num_guassian_images_per_octave):
                if j:
                    octave.append(cv2.GaussianBlur(src=image, ksize=(0, 0), sigmaX=self.sigma**j))
                else:
                    octave.append(image)
            gaussian_images.append(octave)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = [np.stack([cv2.subtract(gaussian_images[j][i], gaussian_images[j][i + 1]) for i in range(self.num_DoG_images_per_octave)]) for j in range(self.num_octaves)]

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []
        for i in range(self.num_octaves):
            h, w = dog_images[i][0].shape
            for j in range(1, self.num_DoG_images_per_octave - 1):
                for p in range(1, h - 1):
                    for q in range(1, w - 1):
                        if abs(dog_images[i][j, p, q]) > self.threshold:
                            block =  dog_images[i][max(0, j - 1):min(j + 2, self.num_DoG_images_per_octave), max(0, p - 1):min(p + 2, h), max(0, q - 1):min(q + 2, w)]
                            if dog_images[i][j, p, q] == np.max(block) or dog_images[i][j, p, q] == np.min(block):
                                keypoints.append([p * (2 ** i), q * (2 ** i)])


        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)


        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
