import numpy as np
import cv2
import copy

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

def pr(connected_matrix):
    pr_im = np.zeros([64, 64], np.str)
    for i in range(connected_matrix.shape[0]):
        for j in range(connected_matrix.shape[1]):
            if connected_matrix[i][j] == 0:
                pr_im[i][j] = 'b'
            elif connected_matrix[i][j] != 0:   
                if i == 0: # top row
                    if j == 0:  # topmost left
                        x2 = 0
                        x3, x0, x1 = 0, connected_matrix[i][j], connected_matrix[i][j + 1]
                        x4 = connected_matrix[i + 1][j]
                    elif j == 63: # topmost right
                        x2 = 0
                        x3, x0, x1 = connected_matrix[i][j - 1], connected_matrix[i][j], 0
                        x4 = connected_matrix[i + 1][j]
                    else:
                        x2 = 0
                        x3, x0, x1 = connected_matrix[i][j - 1], connected_matrix[i][j], connected_matrix[i][j + 1]
                        x4 = connected_matrix[i + 1][j]
                elif i == 63: # last row
                    if j == 0: # bottommost left
                        x2 = connected_matrix[i - 1][j]
                        x3, x0, x1 = 0, connected_matrix[i][j], connected_matrix[i][j + 1]
                        x4 = 0
                    elif j == 63: # bottomost right
                        x2 = connected_matrix[i - 1][j]
                        x3, x0, x1 = connected_matrix[i][j - 1], connected_matrix[i][j], 0
                        x4 = 0
                    else:
                        x2 = connected_matrix[i - 1][j]
                        x3, x0, x1 = connected_matrix[i][j - 1], connected_matrix[i][j], connected_matrix[i][j + 1]
                        x4 = 0
                else: # middle row
                    if j == 0: # leftmost
                        x2 = connected_matrix[i - 1][j]
                        x3, x0, x1 = 0, connected_matrix[i][j], connected_matrix[i][j + 1]
                        x4 = connected_matrix[i + 1][j]
                    elif j == 63: #rightmost
                        x2 = connected_matrix[i - 1][j]
                        x3, x0, x1 = connected_matrix[i][j - 1], connected_matrix[i][j], 0
                        x4 = connected_matrix[i + 1][j]
                    else:
                        x2 = connected_matrix[i - 1][j]
                        x3, x0, x1 = connected_matrix[i][j - 1], connected_matrix[i][j], connected_matrix[i][j + 1]
                        x4 = connected_matrix[i + 1][j]
                h1 = h(x1)
                h2 = h(x2)
                h3 = h(x3)
                h4 = h(x4)
                if h1 + h2 + h3 + h4 < 1 or x0 != 1:
                    pr_im[i][j] = 'q'
                else:
                    pr_im[i][j] = 'p'            
    return pr_im

def h(x):
    if x == 1:
        return 1
    return 0

def shrinking(pr_im, down_im):
    shr_im = np.zeros([64, 64], np.int)
    for i in range(pr_im.shape[0]):
        for j in range(pr_im.shape[1]):
            if pr_im[i][j] == 'q' or pr_im[i][j] == 'b':  # will not be operated
                shr_im[i][j] = down_im[i][j]
            else:  # will be operated(p)
                if i == 0: # top row
                    if j == 0:  # topmost left
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, pr_im[i][j], pr_im[i][j + 1]
                        x8, x4, x5 = 0, pr_im[i + 1][j], pr_im[i + 1][j + 1]
                    elif j == 63: # topmost right
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = pr_im[i][j - 1], pr_im[i][j], 0
                        x8, x4, x5 = pr_im[i + 1][j - 1], pr_im[i + 1][j], 0
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = pr_im[i][j - 1], pr_im[i][j], pr_im[i][j + 1]
                        x8, x4, x5 = pr_im[i + 1][j - 1], pr_im[i + 1][j], pr_im[i + 1][j + 1]
                elif i == 63: # last row
                    if j == 0: # bottommost left
                        x7, x2, x6 = 0, pr_im[i - 1][j], pr_im[i - 1][j + 1]
                        x3, x0, x1 = 0, pr_im[i][j], pr_im[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                    elif j == 63: # bottomost right
                        x7, x2, x6 = pr_im[i - 1][j - 1], pr_im[i - 1][j], 0
                        x3, x0, x1 = pr_im[i][j - 1], pr_im[i][j], 0
                        x8, x4, x5 = 0, 0, 0
                    else:
                        x7, x2, x6 = pr_im[i - 1][j - 1], pr_im[i - 1][j], pr_im[i - 1][j + 1]
                        x3, x0, x1 = pr_im[i][j - 1], pr_im[i][j], pr_im[i][j + 1]
                        x8, x4, x5 = 0, 0, 0
                else: # middle row
                    if j == 0: # leftmost
                        x7, x2, x6 = 0, pr_im[i - 1][j], pr_im[i - 1][j + 1]
                        x3, x0, x1 = 0, pr_im[i][j], pr_im[i][j + 1]
                        x8, x4, x5 = 0, pr_im[i + 1][j], pr_im[i + 1][j + 1]
                    elif j == 63: #rightmost
                        x7, x2, x6 = pr_im[i - 1][j - 1], pr_im[i - 1][j], 0
                        x3, x0, x1 = pr_im[i][j - 1], pr_im[i][j], 0
                        x8, x4, x5 = pr_im[i + 1][j - 1], pr_im[i + 1][j], 0
                    else:
                        x7, x2, x6 = pr_im[i - 1][j - 1], pr_im[i - 1][j], pr_im[i - 1][j + 1]
                        x3, x0, x1 = pr_im[i][j - 1], pr_im[i][j], pr_im[i][j + 1]
                        x8, x4, x5 = pr_im[i + 1][j - 1], pr_im[i + 1][j], pr_im[i + 1][j + 1]
                if x0 == 'b':
                    x0 = 0
                else:
                    x0 = 1
                if x1 == 'b':
                    x1 = 0
                else:
                    x1 = 1
                if x2 == 'b':
                    x2 = 0
                else:
                    x2 = 1
                if x3 == 'b':
                    x3 = 0
                else:
                    x3 = 1
                if x4 == 'b':
                    x4 = 0
                else:
                    x4 = 1
                if x5 == 'b':
                    x5 = 0
                else:
                    x5 = 1
                if x6 == 'b':
                    x6 = 0
                else:
                    x6 = 1
                if x7 == 'b':
                    x7 = 0
                else:
                    x7 = 1
                if x8 == 'b':
                    x8 = 0
                else:
                    x8 = 1
                a1 = shrink_helper(x0, x1, x6, x2)
                a2 = shrink_helper(x0, x2, x7, x3)
                a3 = shrink_helper(x0, x3, x8, x4)
                a4 = shrink_helper(x0, x4, x5, x1)
                if a1 + a2 + a3 + a4 == 1 :
                    shr_im[i][j] = 0 # will turn to black
                    pr_im[i][j] = 'b'
                else:
                    shr_im[i][j] = down_im[i][j] # will not turn to black
    return shr_im

def shrink_helper(b, c, d, e):
    if b == c and (d != b or e != b):
        return 1
    else:
        return 0

im = cv2.imread('lena.bmp', 0)
new_im = binarize(im)
down_im = downsample(new_im)
connected_matrix = yokoi(down_im)  
pr_im = pr(connected_matrix)
shr_im = shrinking(pr_im, down_im)
cv2.imwrite('shrinking1.bmp', shr_im)
i = 1
while ((shr_im != down_im).any()):
    i += 1
    down_im = copy.deepcopy(shr_im)
    connected_matrix = yokoi(shr_im)  
    pr_im = pr(connected_matrix)
    shr_im = shrinking(pr_im, shr_im)
    cv2.imwrite('shrinking%s.bmp'%i, shr_im)
 