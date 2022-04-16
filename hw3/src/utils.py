import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # TODO: 1.forming A
    A = np.zeros((2 * N, 9))
    for i in range(N):
        A[2 * i] = [u[i, 0], u[i, 1], 1, 0, 0, 0, -1 * u[i, 0] * v[i, 0], -1 * u[i, 1] * v[i, 0], -1 * v[i, 0]]
        A[2 * i + 1] = [0, 0, 0, u[i, 0], u[i, 1], 1, -1 * u[i, 0] * v[i, 1], -1 * u[i, 1] * v[i, 1], -1 * v[i, 1]]

    # TODO: 2.solve H with A
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape((3, 3))

    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    dst_coor = np.stack(np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax)), axis=-1).reshape((-1, 2))
    N = dst_coor.shape[0]

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    dst_coor = np.concatenate((dst_coor, np.ones((N, 1))), axis=1)

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_coor = H_inv @ dst_coor.T
        new_coor = np.rint(new_coor / new_coor[-1]).astype(int)
        new_coor = new_coor[:-1].reshape((2, (ymax - ymin), (xmax - xmin)))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (new_coor[0] > 0) * (new_coor[1] > 0) * (new_coor[0] < w_src) * (new_coor[1] < h_src)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        dst_coor = dst_coor.reshape(((ymax - ymin), (xmax - xmin), -1)).astype(int)

        # TODO: 6. assign to destination image with proper masking
        dst[dst_coor[:, :, 1][mask], dst_coor[:, :, 0][mask]] = src[new_coor[1][mask], new_coor[0][mask]]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_coor = H @ dst_coor.T
        new_coor = np.rint(new_coor / new_coor[-1]).astype(int)
        new_coor = new_coor[:-1].reshape((2, (ymax - ymin), (xmax - xmin)))

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        # TODO: 5.filter the valid coordinates using previous obtained mask

        # TODO: 6. assign to destination image using advanced array indicing
        dst[new_coor[1].clip(min=0, max=h_dst - 1), new_coor[0].clip(min=0, max=w_dst - 1)] = src

    return dst
