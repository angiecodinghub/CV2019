import numpy as np
import cv2
import matplotlib.pyplot as plt

def binarize(im):
    new_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i][j] >= 128:
                new_im[i][j] = 255
            else:
                new_im[i][j] = 0
    return new_im

def downsample(im):
    down_im = np.zeros([64, 64], np.int)
    for i in range(0, im.shape[0], 8):
        for j in range(0, im.shape[1], 8):
            down_im[(i)//8][(j)//8] = im[i][j]
    return down_im

def yokoi(down_im):
    connected_matrix = np.zeros([64, 64], np.int)
    for i in range(down_im.shape[0]):
        for j in range(down_im.shape[1]):
            if down_im[i][j] > 0:
                if i == 0: # top row
                    if j == 0:  # topmost left
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, down_im[i][j], down_im[i][j + 1]
                        x8, x4, x5 = 0, down_im[i + 1][j], down_im[i + 1][j + 1]
                    elif j == 63: # topmost right
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = down_im[i][j - 1], down_im[i][j], 0
                        x8, x4, x5 = down_im[i + 1][j - 1], down_im[i + 1][j], 0
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = down_im[i][j - 1], down_im[i][j], down_im[i][j + 1]
                        x8, x4, x5 = down_im[i + 1][j - 1], down_im[i + 1][j], down_im[i + 1][j + 1]
                elif i == 63: # last row
                    if j == 0: # bottommost left
                        x7, x2, x6 = 0, down_im[i - 1][j], down_im[i - 1][j + 1]
                        x3, x0, x1 = 0, down_im[i][j], down_im[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == 63: # bottomost right
                        x7, x2, x6 = down_im[i - 1][j - 1], down_im[i - 1][j], 0
                        x3, x0, x1 = down_im[i][j - 1], down_im[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                        x7, x2, x6 = down_im[i - 1][j - 1], down_im[i - 1][j], down_im[i - 1][j + 1]
                        x3, x0, x1 = down_im[i][j - 1], down_im[i][j], down_im[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                else: # middle row
                    if j == 0: # leftmost
                        x7, x2, x6 = 0, down_im[i - 1][j], down_im[i - 1][j + 1]
                        x3, x0, x1 = 0, down_im[i][j], down_im[i][j + 1]
                        x8, x4, x5 = 0, down_im[i + 1][j], down_im[i + 1][j + 1]
                    elif j == 63: #rightmost
                        x7, x2, x6 = down_im[i - 1][j - 1], down_im[i - 1][j], 0
                        x3, x0, x1 = down_im[i][j - 1], down_im[i][j], 0
                        x8, x4, x5 = down_im[i + 1][j - 1], down_im[i + 1][j], 0
                    else:
                        x7, x2, x6 = down_im[i - 1][j - 1], down_im[i - 1][j], down_im[i - 1][j + 1]
                        x3, x0, x1 = down_im[i][j - 1], down_im[i][j], down_im[i][j + 1]
                        x8, x4, x5 = down_im[i + 1][j - 1], down_im[i + 1][j], down_im[i + 1][j + 1]
                a1 = yokoi_func(x0, x1, x6, x2)
                a2 = yokoi_func(x0, x2, x7, x3)
                a3 = yokoi_func(x0, x3, x8, x4)
                a4 = yokoi_func(x0, x4, x5, x1)
                if a1 == a2 == a3 == a4 == 'r':
                    connected_matrix[i][j] = 5
                else:
                    count = 0
                    for l in [a1, a2, a3, a4]:
                        if l == 'q':
                            count += 1
                        if count > 0:
                            connected_matrix[i][j] = count
                        else:
                            connected_matrix[i][j] = 0
    return connected_matrix

def yokoi_func(b, c, d, e):
    if b == c and (d != b or e != b):
        return 'q'
    if b == c and (d == b and e == b):
        return 'r'
    return 's'
            
im = cv2.imread('lena.bmp', 0)
new_im = binarize(im)
down_im = downsample(new_im)
connected_matrix = yokoi(down_im)
with open('result.txt', 'w') as f:
    for row in connected_matrix:
        print(''.join(map(str,row)), file=f)
        