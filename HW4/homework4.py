import numpy as np
import cv2

def binarize(im):
    new_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] >= 128:
                new_im[i][j] = 255
            else:
                new_im[i][j] = 0
    return new_im

def dilation(im, kernel):
    dil_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] > 0:
                for ind in kernel:
                    p, q = ind
                    if (i + p) >= 0 and (j + q) >= 0 and  (i + p) <= (im.shape[0] - 1) and (j + q) <= (im.shape[1] - 1):
                        dil_im[i + p][j + q] = 255
    return dil_im

def erosion(im, kernel):
    ero_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] > 0:
                save = True
                for ind in kernel:
                    p, q = ind
                    if (i + p) >= 0 and (j + q) >= 0 and (i + p) <= (im.shape[0] - 1) and (j + q) <= (im.shape[1] - 1):
                        if im[i + p][j + q] != 255:
                            save = False
                            break
                    else:
                        save = False
                        break
                if save:
                    ero_im[i][j] = 255
    return ero_im

def erosion_hm2(im, kernel):
    ero_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] >= 0:
                save = True
                for ind in kernel:
                    p, q = ind
                    if (i + p) >= 0 and (j + q) >= 0 and (i + p) <= (im.shape[0] - 1) and (j + q) <= (im.shape[1] - 1):
                        if im[i + p][j + q] != 255:
                            save = False
                            break
                    else:
                        save = False
                        break
                if save:
                    ero_im[i][j] = 255
    return ero_im

def opening(im, kernel):
    return dilation(erosion(im, kernel), kernel)

def closing(im, kernel):
    return erosion(dilation(im, kernel), kernel)

def hit_and_miss(im, kernel_ham1, kernel_ham2):
    reverse_im = np.zeros(im.shape, np.int)
    ham_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] == 255:
               reverse_im[i][j] = 0
            else:
                reverse_im[i][j] = 255
    a_im = erosion(im, kernel_ham1)
    b_im = erosion_hm2(reverse_im, kernel_ham2)
    for i in range(a_im.shape[0]):
        for j in range(a_im.shape[1]):
            if a_im[i][j] == 255 and b_im[i][j] == 255:
                ham_im[i][j] = 255
    return ham_im

im = cv2.imread('lena.bmp', 0)
binarized = binarize(im)
kernel = [[-2, -1], [-2, 0], [-2, 1],
          [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
          [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
          [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
          [2, -1], [2, 0], [2, 1]]
kernel_ham1 = [[0, -1], [0, 0], [1, 0]]
kernel_ham2 = [[-1, 0], [-1, 1], [0, 1]]
dil_im = dilation(binarized, kernel)
cv2.imwrite('dilation.bmp', dil_im)
ero_im = erosion(binarized, kernel)
cv2.imwrite('erosion.bmp', ero_im)
open_im = opening(binarized, kernel)
cv2.imwrite('opening.bmp', open_im)
close_im = closing(binarized, kernel)
cv2.imwrite('closing.bmp', close_im)
ham_im = hit_and_miss(binarized, kernel_ham1, kernel_ham2)
cv2.imwrite('hit_and_miss.bmp', ham_im)