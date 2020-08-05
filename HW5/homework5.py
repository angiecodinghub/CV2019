import numpy as np
import cv2

def dilation(im, kernel):
    dil_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] > 0:
                max_v = 0
                for ind in kernel:
                    p, q = ind
                    if (i + p) >= 0 and (j + q) >= 0 and (i + p) <= (im.shape[0] - 1) and (j + q) <= (im.shape[1] - 1):
                        if im[i + p][j + q] > max_v:
                            max_v = im[i + p][j + q]
                for ind in kernel:
                    p, q = ind
                    if (i + p) >= 0 and (j + q) >= 0 and (i + p) <= (im.shape[0] - 1) and (j + q) <= (im.shape[1] - 1):
                        dil_im[i + p][j + q] = max_v
    return dil_im

def erosion(im, kernel):
    ero_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] > 0:
                min_v = 10000
                for ind in kernel:
                    p, q = ind
                    if (i + p) >= 0 and (j + q) >= 0 and (i + p) <= (im.shape[0] - 1) and (j + q) <= (im.shape[1] - 1):
                        if im[i + p][j + q] < min_v:
                            min_v = im[i + p][j + q]
                for ind in kernel:
                    p, q = ind
                    if (i + p) >= 0 and (j + q) >= 0 and (i + p) <= (im.shape[0] - 1) and (j + q) <= (im.shape[1] - 1):
                        ero_im[i + p][j + q] = min_v
    return ero_im

def opening(im, kernel):
    return dilation(erosion(im, kernel), kernel)

def closing(im, kernel):
    return erosion(dilation(im, kernel), kernel)

im = cv2.imread('lena.bmp', 0)
kernel = [[-2, -1], [-2, 0], [-2, 1],
         [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
         [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
         [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
         [2, -1], [2, 0], [2, 1]]
im_dil = dilation(im, kernel)
cv2.imwrite('dilation.bmp', im_dil)
ero_im = erosion(im, kernel)
cv2.imwrite('erosion.bmp', ero_im)
open_im = opening(im, kernel)
cv2.imwrite('opening.bmp', open_im)
close_im = closing(im, kernel)
cv2.imwrite('closing.bmp', close_im)