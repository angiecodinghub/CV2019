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

def histogram_data(im):
    count_data = np.zeros(256, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            count_data[im[i][j]] += 1
    np.savetxt("count_data.csv", count_data, delimiter=",")

def connected_components(binar_im):
    binarized = np.zeros(binar_im.shape, np.int)
    area = 1
    for i in range(binar_im.shape[0]):
        for j in range(binar_im.shape[1]):
            if binar_im[i][j] > 0:
                binarized[i][j] = area
                area += 1
    binarized_backup = copy.deepcopy(binarized)
    while True:
        # top-down
        for i in range(binarized.shape[0]):
            for j in range(binarized.shape[1]):
                if binarized[i][j] != 0:
                    if i == 0 and j == 0:
                        binarized[i][j] = binarized[i][j]
                    elif i == 0:
                        a = np.array([binarized[i][j],binarized[i][j-1]])
                        binarized[i][j] = np.min(a[np.nonzero(a)])
                    elif j == 0:
                        a = np.array([binarized[i][j],binarized[i-1][j]])
                        binarized[i][j] = np.min(a[np.nonzero(a)])
                    else:
                        a = np.array([binarized[i][j],binarized[i-1][j],binarized[i][j-1]])
                        binarized[i][j] = np.min(a[np.nonzero(a)])  
        if False not in (binarized == binarized_backup):
            break
        binarized_backup = copy.deepcopy(binarized)
        #buttom-up
        for i in range(binarized.shape[0]-1, -1, -1):
            for j in range(binarized.shape[1]-1, -1, -1):
                if binarized[i][j] != 0:
                    if i == binarized.shape[0]-1 and j == binarized.shape[0]-1:
                        binarized[i][j] = binarized[i][j]
                    elif i == binarized.shape[0]-1:
                        a = np.array([binarized[i][j],binarized[i][j+1]])
                        binarized[i][j] = np.min(a[np.nonzero(a)])
                    elif j == binarized.shape[0]-1:
                        a = np.array([binarized[i][j],binarized[i+1][j]])
                        binarized[i][j] = np.min(a[np.nonzero(a)])
                    else:
                        a = np.array([binarized[i][j],binarized[i+1][j],binarized[i][j+1]])
                        binarized[i][j] = np.min(a[np.nonzero(a)])
        if False not in (binarized == binarized_backup):
            break
        binarized_backup = copy.deepcopy(binarized)
    # mute the ones < 500
    count_pixel = np.zeros(np.max(binarized) + 1, np.int)
    for i in range(binarized.shape[0]):
        for j in range(binarized.shape[1]):
            if binarized[i][j] > 0:
                count_pixel[binarized[i][j]] += 1
    for i in range(binarized.shape[0]):
        for j in range(binarized.shape[1]):
            if count_pixel[binarized[i][j]] < 500:
                binarized[i][j] = 0
    # plot the bounding box
    for i in range(1, count_pixel.shape[0]):
        if count_pixel[i] >= 500:
            ind_set = np.array(np.where(binarized == i)).T
            ind_up_left = np.array([np.min(ind_set[:, 0]), np.min(ind_set[:, 1])])
            ind_bottom_right = np.array([np.max(ind_set[:, 0]), np.max(ind_set[:, 1])])
            # draw the rectangle
            cv2.rectangle(binar_im, (ind_up_left[1], ind_up_left[0]), (ind_bottom_right[1], ind_bottom_right[0]),(139,0,139), 3)
            # draw the middle points
            # centroid(x)
            upper = 0
            area = 0
            for i in range(ind_up_left[1], ind_bottom_right[1]+1):
                upper_plus = 0
                for j in range(ind_up_left[0], ind_bottom_right[0]+1):
                    if binarized[j][i]!= 0:
                        upper_plus += 1
                        area += 1
                upper += i * upper_plus
            first = upper // area
            # centroid(y)
            upper = 0
            area = 0
            for i in range(ind_up_left[0], ind_bottom_right[0]+1):
                upper_plus = 0
                for j in range(ind_up_left[1], ind_bottom_right[1]+1):
                    if binarized[i][j]!= 0:
                        upper_plus += 1
                        area += 1
                upper += i * upper_plus
            second = upper // area
            '''
            first = (ind_up_left[1] + ind_bottom_right[1]) // 2
            second = (ind_up_left[0] + ind_bottom_right[0]) // 2
            '''
            cv2.circle(binar_im, (first, second), 3, (139, 0, 139), 3)
            
    cv2.imwrite('connect_4.bmp', binar_im)   
    
im = cv2.imread('lena.bmp', 0)
new_im_1 = binarize(im)
cv2.imwrite('binarize.bmp', new_im_1)
histogram_data(im)
binar_im = cv2.imread('binarize.bmp', 0)
connected_components(binar_im)