import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    with open(args.setting_path, 'r') as f:
        lines = f.readlines()
    sigma = lines[6].split(',')
    JBF = Joint_bilateral_filter(int(sigma[1]), float(sigma[3]))
    bf_out = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    for i in range(1, 6):
        lines[i] = [float(j) for j in lines[i].split(',')]
    rgb = np.array(lines[1:6])
    gray = np.concatenate((img_gray[:,:,None], np.matmul(img_rgb, rgb.T)), axis=-1)
    costs = []
    for i in range(gray.shape[-1]):
        jbf_out = JBF.joint_bilateral_filter(img_rgb, gray[:,:,i]).astype(np.uint8)
        costs.append((np.sum(np.abs(bf_out.astype(np.int32) - jbf_out.astype(np.int32))), cv2.cvtColor(jbf_out, cv2.COLOR_RGB2BGR), gray[:,:,i]))
    print('cost:', [c[0] for c in costs])
    max_cost = max(costs)
    min_cost = min(costs)

    IMG_NAME = os.path.splitext(os.path.basename(args.image_path))[0]
    FILTERED_PATH = 'filtered_rgb'
    GRAY_PATH = 'gray'
    os.makedirs(FILTERED_PATH, exist_ok=True)
    os.makedirs(GRAY_PATH, exist_ok=True)
    cv2.imwrite(os.path.join(FILTERED_PATH, f'{IMG_NAME}_max.png'), max_cost[1])
    cv2.imwrite(os.path.join(GRAY_PATH, f'{IMG_NAME}_max.png'), max_cost[2])
    cv2.imwrite(os.path.join(FILTERED_PATH, f'{IMG_NAME}_min.png'), min_cost[1])
    cv2.imwrite(os.path.join(GRAY_PATH, f'{IMG_NAME}_min.png'), min_cost[2])


if __name__ == '__main__':
    main()