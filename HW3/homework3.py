import numpy as np
import cv2
import matplotlib.pyplot as plt

def histogram_data(im, num):
    count_data = np.zeros(256, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            count_data[im[i][j]] += 1
    fig = plt.figure()
    plt.bar(np.arange(257), np.append(count_data, np.array([1])), color='b')
    plt.xlabel('pixel value')
    plt.ylabel('number')
    if num == 1:
        plt.title('original histogram')
        plt.savefig('histogram_original.png')
    elif num == 2:
        plt.title('divided by 3 histogram')
        plt.savefig('histogram_div_3.png')
    elif num == 3:
        plt.title('equalized histogram')
        plt.savefig('histogram_equalization.png')

def divide_by_three(im):
    new_im = np.zeros(im.shape, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
                new_im[i][j] = im[i][j] // 3
    return new_im

def equalization(im):
    new_im = np.zeros(im.shape, np.int)  
    count_data = np.zeros(256, np.int)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            count_data[im[i][j]] += 1
    new_data = np.zeros(256, np.int)
    new_data[0] = count_data[0]
    for i in range(1, 256):
        new_data[i] = new_data[i - 1] + count_data[i]
    new_data = new_data * 255 / np.sum(count_data)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            new_im[i][j] = new_data[im[i][j]]
    return new_im
  
im = cv2.imread('lena.bmp', 0)
histogram_data(im, 1)
im_div_3 = divide_by_three(im)
cv2.imwrite('div_3.bmp', im_div_3)
histogram_data(im_div_3, 2)
equalized = equalization(im_div_3)
cv2.imwrite('equalized.bmp', equalized)
histogram_data(equalized, 3)