import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s

        w = np.arange(self.wndw_size)
        xx, yy = np.meshgrid(w, w)
        self.Gs = np.exp(
            ((self.pad_w - xx) ** 2 + (self.pad_w - yy) ** 2)/(-2 * (self.sigma_s ** 2)))

        self.table = np.exp(((np.arange(256) / 255) ** 2) /
                            (-2 * (self.sigma_r ** 2)))

    def joint_bilateral_filter(self, img, guidance, improved=True):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        H, W, C = img.shape
        output = np.zeros((H, W, C))
        if improved:
            if padded_guidance.ndim < 3:
                padded_guidance = padded_guidance.reshape(
                    padded_guidance.shape[0], padded_guidance.shape[1], 1)
            for i in range(self.wndw_size):
                for j in range(self.wndw_size):
                    Tp = padded_guidance[self.pad_w + i:H + self.pad_w:self.wndw_size,
                                         self.pad_w + j:W + self.pad_w:self.wndw_size]
                    Tq = padded_guidance[i:Tp.shape[0] * self.wndw_size + i, j:Tp.shape[1] * self.wndw_size + j].reshape(
                        Tp.shape[0], self.wndw_size, Tp.shape[1], self.wndw_size, -1).swapaxes(1, 2).swapaxes(3, 4).swapaxes(2, 3)
                    Gr = cv2.LUT(
                        np.abs(Tp[:, :, :, None, None] - Tq).astype(np.uint8), self.table)
                    if Gr.ndim > 2:
                        Gr = Gr.prod(axis=2)
                    G = self.Gs * Gr
                    output[i::self.wndw_size, j::self.wndw_size] = (G[:, :, :, :, None] * padded_img[i:Tp.shape[0] * self.wndw_size + i, j:Tp.shape[1] * self.wndw_size + j].reshape(
                        Tp.shape[0], self.wndw_size, Tp.shape[1], self.wndw_size, -1).swapaxes(1, 2)).sum(axis=(2, 3)) / G.sum(axis=(2, 3)).reshape(Tp.shape[0], Tp.shape[1], 1)
        else:
            for i in range(self.pad_w, H + self.pad_w):
                for j in range(self.pad_w, W + self.pad_w):
                    Tp = padded_guidance[i, j] / 255
                    Tq = padded_guidance[i - self.pad_w:i +
                                         self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1] / 255
                    e = ((Tp - Tq) ** 2)/(-2 * (self.sigma_r ** 2))
                    if e.ndim > 2:
                        e = e.sum(axis=-1)
                    Gr = np.exp(e)
                    G = self.Gs * Gr
                    output[i - self.pad_w, j - self.pad_w] = (G[:, :, None] * padded_img[i - self.pad_w:i +
                                                              self.pad_w + 1, j - self.pad_w:j + self.pad_w + 1]).sum(axis=(0, 1)) / G.sum()

        return np.clip(output, 0, 255).astype(np.uint8)
