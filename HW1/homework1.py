import numpy as np
import cv2

def upside_down(im):
    new_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        new_im[i, :] = im[im.shape[0] - i - 1, :]
    return new_im

def right_side_left(im):
    new_im = np.zeros(im.shape, np.int)
    for j in range(im.shape[1]):
        new_im[:, j] = im[:, im.shape[1] - j - 1]
    return new_im

def diagonal(im):
    new_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(i):
            new_im[i, j] = im[j, i]
            new_im[j, i] = im[j, i]
            new_im[j, j] = im[j, j]
    return new_im

im = cv2.imread('lena.bmp')
new_im_1 = upside_down(im)
cv2.imwrite('upside_down.bmp', new_im_1)
new_im_2 = right_side_left(im)
cv2.imwrite('right_side_left.bmp', new_im_2)
new_im_3 = diagonal(im)
cv2.imwrite('diagonally_mirrored.bmp', new_im_3)