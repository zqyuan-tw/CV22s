import numpy as np
import cv2.ximgproc as xip
import cv2


def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    left_cost, right_cost = cost_matching(Il, Ir, max_disp)


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    smooth_left, smooth_right = cost_aggregation(Il, Ir, left_cost, right_cost, max_disp)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    left_disp = smooth_left.argmin(-1)
    right_disp = smooth_right.argmin(-1)

    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    hole = max_disp + 1
    left_disp = left_right_consistency_checking(left_disp, right_disp, hole)
    labels = hole_filling(left_disp, max_disp, hole)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels.astype(np.uint8), 15, 1)


    return labels.astype(np.uint8)

def hamming_space(img):
    pad_img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    window = np.stack((pad_img[0:-2, 0:-2], pad_img[0:-2, 1:-1], pad_img[0:-2, 2:], pad_img[1:-1, 0:-2], pad_img[1:-1, 2:], pad_img[2:, 0:-2], pad_img[2:, 1:-1], pad_img[2:, 2:]), axis=-1)

    return img[:, :, :, np.newaxis] > window

def cost_matching(left, right, max_disp):
    left_hamming, right_hamming = hamming_space(left), hamming_space(right)
    H, W, C, L = left_hamming.shape 
    left_cost = np.zeros((H, W, max_disp + 1), dtype=np.float32)
    right_cost = np.zeros((H, W, max_disp + 1), dtype=np.float32)
    for d in range(max_disp + 1):
        l, r = left_hamming[:, d:], right_hamming[:, :W - d]        
        hamming_dist = np.sum(l != r, axis=(2, 3))
        pad_left, pad_right = np.tile(hamming_dist[:, 0:1], (1, d)), np.tile(hamming_dist[:, -1:], (1, d))
        left_cost[:, :, d] = np.hstack((pad_left, hamming_dist))
        right_cost[:, :, d] = np.hstack((hamming_dist, pad_right))
        
    return left_cost, right_cost

def cost_aggregation(left, right, left_cost, right_cost, max_disp):
    H, W, C = left.shape 
    smooth_left = np.zeros((H, W, max_disp + 1))
    smooth_right = np.zeros((H, W, max_disp + 1))
    for d in range(max_disp+1):
        smooth_left[:, :, d] = xip.jointBilateralFilter(left, left_cost[:, :, d], 0, 15, 5)
        smooth_right[:, :, d] = xip.jointBilateralFilter(right, right_cost[:, :, d], 0, 15, 5)
    
    return smooth_left, smooth_right

def left_right_consistency_checking(left_disp, right_disp, hole):
    H, W = left_disp.shape
    for h in range(H):
        for w in range(W):
            d =left_disp[h, w]
            if w >= d and d != right_disp[h, w - d]:
                left_disp[h, w] = hole
                
    return left_disp

def hole_filling(disp, max_disp, hole):
    pad_disp = np.pad(disp, ((0,0), (1,1)), 'constant', constant_values=max_disp)
    H, W = pad_disp.shape
    FL, FR = np.copy(pad_disp), np.copy(pad_disp)
    for h in range(H):
        for w in range(1, W - 1):
            if pad_disp[h, w] == hole:
                left_bound = w - 1
                while pad_disp[h, left_bound] == hole and left_bound:
                    left_bound = left_bound - 1
                right_bound = w + 1
                while pad_disp[h, right_bound] == hole and right_bound < W:
                    right_bound = right_bound + 1
                FL[h, w], FR[h, w] = pad_disp[h, left_bound], pad_disp[h, right_bound] 
    labels = np.minimum(FL, FR)

    return labels[:, 1:-1]
    