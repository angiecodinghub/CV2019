import numpy as np
import cv2
import copy
import math

def gaussian_noise(im, amp):
    return im + amp * np.random.normal(0, 1, im.shape)

def sap_noise(im, threshold):
    probability = np.random.uniform(0, 1, im.shape)
    sap_im = copy.deepcopy(im)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if probability[i][j] < threshold:
                sap_im[i][j] = 0
            elif probability[i][j] > 1 - threshold:
                sap_im[i][j] = 255
    return sap_im

def box_filter(im, filtersize):
    box_im = np.zeros(im.shape, np.int)
    if filtersize == 3:
        ope_im = cv2.copyMakeBorder(im, 1, 1, 1, 1,cv2.BORDER_REFLECT)
        for i in range(1, ope_im.shape[0] - 1):
            for j in range(1, ope_im.shape[1] - 1):
                box_im[i-1][j-1] = np.mean(ope_im[i-1:i+filtersize-1, j-1:j+filtersize-1])
    elif filtersize == 5:
        ope_im = cv2.copyMakeBorder(im, 2, 2, 2, 2,cv2.BORDER_REFLECT)
        for i in range(2, ope_im.shape[0] - 2):
            for j in range(2, ope_im.shape[1] - 2):
                box_im[i-2][j-2] = np.mean(ope_im[i-2:i+filtersize-2, j-2:j+filtersize-2])
    return box_im

def median_filter(im, filtersize):
    med_im = np.zeros(im.shape, np.int)
    if filtersize == 3:
        ope_im = cv2.copyMakeBorder(im, 1, 1, 1, 1,cv2.BORDER_REFLECT)
        for i in range(1, ope_im.shape[0] - 1):
            for j in range(1, ope_im.shape[1] - 1):
                med_im[i-1][j-1] = np.median(ope_im[i-1:i+filtersize-1, j-1:j+filtersize-1])
    elif filtersize == 5:
        ope_im = cv2.copyMakeBorder(im, 2, 2, 2, 2,cv2.BORDER_REFLECT)
        for i in range(2, ope_im.shape[0] - 2):
            for j in range(2, ope_im.shape[1] - 2):
                med_im[i-2][j-2] = np.median(ope_im[i-2:i+filtersize-2, j-2:j+filtersize-2])
    return med_im

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

def SNR(signal_im, noise_im):
    signal_mu = 0
    noise_mu = 0
    signal_v = 0
    noise_v = 0
    for i in range(signal_im.shape[0]):
        for j in range(signal_im.shape[1]):
            if signal_im[i][j] == 255:
                signal_im[i][j] = 1
    for i in range(noise_im.shape[0]):
        for j in range(noise_im.shape[1]):
            if noise_im[i][j] == 255:
                noise_im[i][j] = 1
    for i in range(signal_im.shape[0]):
        for j in range(signal_im.shape[1]):
            signal_mu += int(signal_im[i][j])
    signal_mu = signal_mu / (signal_im.shape[0] * signal_im.shape[1])
    for i in range(noise_im.shape[0]):
        for j in range(noise_im.shape[1]):
            noise_mu += (int(noise_im[i][j]) - int(signal_im[i][j]))
    noise_mu = noise_mu / (noise_im.shape[0] * noise_im.shape[1])
    for i in range(signal_im.shape[0]):
        for j in range(signal_im.shape[1]):
            signal_v += math.pow(int(signal_im[i][j]) - signal_mu, 2)
    signal_v = signal_v / (signal_im.shape[0] * signal_im.shape[1])
    for i in range(noise_im.shape[0]):
        for j in range(noise_im.shape[1]):
            noise_v += math.pow(int(noise_im[i][j]) - int(signal_im[i][j]) - noise_mu, 2)
    noise_v = noise_v / (noise_im.shape[0] * noise_im.shape[1])
    for i in range(signal_im.shape[0]):
        for j in range(signal_im.shape[1]):
            if signal_im[i][j] == 1:
                signal_im[i][j] = 255
    for i in range(noise_im.shape[0]):
        for j in range(noise_im.shape[1]):
            if noise_im[i][j] == 1:
                noise_im[i][j] = 255
    return 20 * math.log(math.sqrt(signal_v) / math.sqrt(noise_v), 10)
im = cv2.imread('lena.bmp', 0)
gaussi_10 = gaussian_noise(im, 10)
gaussi_10_SNR = SNR(im, gaussi_10)
cv2.imwrite('gaussi_10.bmp', gaussi_10)
gaussi_30 = gaussian_noise(im, 30)
gaussi_30_SNR = SNR(im, gaussi_30)
cv2.imwrite('gaussi_30.bmp', gaussi_30)
sap_im_05 = sap_noise(im, 0.05)
sap_im_05_SNR = SNR(im, sap_im_05)
cv2.imwrite('sap_im_05.bmp', sap_im_05)
sap_im_10 = sap_noise(im, 0.10)
sap_im_10_SNR = SNR(im, sap_im_10)
cv2.imwrite('sap_im_10.bmp', sap_im_10)
######### BOX fILTER #########
box_g_10_3 = box_filter(gaussi_10, 3)
box_g_10_3_SNR = SNR(im, box_g_10_3)
cv2.imwrite('box_g_10_3.bmp', box_g_10_3)
box_g_10_5 = box_filter(gaussi_10, 5)
box_g_10_5_SNR = SNR(im, box_g_10_5)
cv2.imwrite('box_g_10_5.bmp', box_g_10_5)
box_g_30_3 = box_filter(gaussi_30, 3)
box_g_30_3_SNR = SNR(im, box_g_30_3)
cv2.imwrite('box_g_30_3.bmp', box_g_30_3)
box_g_30_5 = box_filter(gaussi_30, 5)
box_g_30_5_SNR = SNR(im, box_g_30_5)
cv2.imwrite('box_g_30_5.bmp', box_g_30_5)
box_s_05_3 = box_filter(sap_im_05, 3)
box_s_05_3_SNR = SNR(im, box_s_05_3)
cv2.imwrite('box_s_05_3.bmp', box_s_05_3)
box_s_05_5 = box_filter(sap_im_05, 5)
box_s_05_5_SNR = SNR(im, box_s_05_5)
cv2.imwrite('box_s_05_5.bmp', box_s_05_5)
box_s_10_3 = box_filter(sap_im_10, 3)
box_s_10_3_SNR = SNR(im, box_s_10_3)
cv2.imwrite('box_s_10_3.bmp', box_s_10_3)
box_s_10_5 = box_filter(sap_im_10, 5)
box_s_10_5_SNR = SNR(im, box_s_10_5)
cv2.imwrite('box_s_10_5.bmp', box_s_10_5)
######### MEDIAN FILTER #########
med_g_10_3 = median_filter(gaussi_10, 3)
med_g_10_3_SNR = SNR(im, med_g_10_3)
cv2.imwrite('med_g_10_3.bmp', med_g_10_3)
med_g_10_5 = median_filter(gaussi_10, 5)
med_g_10_5_SNR = SNR(im, med_g_10_5)
cv2.imwrite('med_g_10_5.bmp', med_g_10_5)
med_g_30_3 = median_filter(gaussi_30, 3)
med_g_30_3_SNR = SNR(im, med_g_30_3)
cv2.imwrite('med_g_30_3.bmp', med_g_30_3)
med_g_30_5 = median_filter(gaussi_30, 5)
med_g_30_5_SNR = SNR(im, med_g_30_5)
cv2.imwrite('med_g_30_5.bmp', med_g_30_5)
med_s_05_3 = median_filter(sap_im_05, 3)
med_s_05_3_SNR = SNR(im, med_s_05_3)
cv2.imwrite('med_s_05_3.bmp', med_s_05_3)
med_s_05_5 = median_filter(sap_im_05, 5)
med_s_05_5_SNR = SNR(im, med_s_05_5)
cv2.imwrite('med_s_05_5.bmp', med_s_05_5)
med_s_10_3 = median_filter(sap_im_10, 3)
med_s_10_3_SNR = SNR(im, med_s_10_3)
cv2.imwrite('med_s_10_3.bmp', med_s_10_3)
med_s_10_5 = median_filter(sap_im_10, 5)
med_s_10_5_SNR = SNR(im, med_s_10_5)
cv2.imwrite('med_s_10_5.bmp', med_s_10_5)
######### OPENING THEN CLOSING #########
kernel = [[-2, -1], [-2, 0], [-2, 1],
         [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
         [0, -2], [0, -1], [0, 0], [0, 1], [0, 2],
         [1, -2], [1, -1], [1, 0], [1, 1], [1, 2],
         [2, -1], [2, 0], [2, 1]]
otc_g_10 = closing(opening(gaussi_10, kernel), kernel)
otc_g_10_SNR = SNR(im, otc_g_10)
cv2.imwrite('otc_g_10.bmp', otc_g_10)
otc_g_30 = closing(opening(gaussi_30, kernel), kernel)
otc_g_30_SNR = SNR(im, otc_g_30)
cv2.imwrite('otc_g_30.bmp', otc_g_30)
otc_s_05 = closing(opening(sap_im_05, kernel), kernel)
otc_s_05_SNR = SNR(im, otc_s_05)
cv2.imwrite('otc_s_05.bmp', otc_s_05)
otc_s_10 = closing(opening(sap_im_10, kernel), kernel)
otc_s_10_SNR = SNR(im, otc_s_10)
cv2.imwrite('otc_s_10.bmp', otc_s_10)
######### CLOSING THEN OPENING #########
cto_g_10 = opening(closing(gaussi_10, kernel), kernel)
cto_g_10_SNR = SNR(im, cto_g_10)
cv2.imwrite('cto_g_10.bmp', cto_g_10)
cto_g_30 = opening(closing(gaussi_30, kernel), kernel)
cto_g_30_SNR = SNR(im, cto_g_30)
cv2.imwrite('cto_g_30.bmp', cto_g_30)
cto_s_05 = opening(closing(sap_im_05, kernel), kernel)
cto_s_05_SNR = SNR(im, cto_s_05)
cv2.imwrite('cto_s_05.bmp', cto_s_05)
cto_s_10 = opening(closing(sap_im_10, kernel), kernel)
cto_s_10_SNR = SNR(im, cto_s_10)
cv2.imwrite('cto_s_10.bmp', cto_s_10)

file = open("SNR.txt", "w")
file.write("gaussi_10_SNR: " + str(gaussi_10_SNR) + '\n')
file.write("gaussi_30_SNR " + str(gaussi_30_SNR) + '\n')
file.write("sap_im_05_SNR: " + str(sap_im_05_SNR) + '\n')
file.write("sap_im_10_SNR: " + str(sap_im_10_SNR) + '\n')
file.write("box_g_10_3_SNR: " + str(box_g_10_3_SNR) + '\n')
file.write("box_g_10_5_SNR: " + str(box_g_10_5_SNR) + '\n')
file.write("box_g_30_3_SNR: " + str(box_g_30_3_SNR) + '\n')
file.write("box_g_30_5_SNR: " + str(box_g_30_5_SNR) + '\n')
file.write("box_s_05_3_SNR: " + str(box_s_05_3_SNR) + '\n')
file.write("box_s_05_5_SNR: " + str(box_s_05_5_SNR) + '\n')
file.write("box_s_10_3_SNR: " + str(box_s_10_3_SNR) + '\n')
file.write("box_s_10_5_SNR: " + str(box_s_10_5_SNR) + '\n')
file.write("med_g_10_3_SNR: " + str(med_g_10_3_SNR) + '\n')
file.write("med_g_10_5_SNR: " + str(med_g_10_5_SNR) + '\n')
file.write("med_g_30_3_SNR: " + str(med_g_30_3_SNR) + '\n')
file.write("med_g_30_5_SNR: " + str(med_g_30_5_SNR) + '\n')
file.write("med_s_05_3_SNR: " + str(med_s_05_3_SNR) + '\n')
file.write("med_s_05_5_SNR: " + str(med_s_05_5_SNR) + '\n')
file.write("med_s_10_3_SNR: " + str(med_s_10_3_SNR) + '\n')
file.write("med_s_10_5_SNR: " + str(med_s_10_5_SNR) + '\n')
file.write("otc_g_10_SNR: " + str(otc_g_10_SNR) + '\n')
file.write("otc_g_30_SNR: " + str(otc_g_30_SNR) + '\n')
file.write("otc_s_05_SNR: " + str(otc_s_05_SNR) + '\n')
file.write("otc_s_10_SNR: " + str(otc_s_10_SNR) + '\n')
file.write("cto_g_10_SNR: " + str(cto_g_10_SNR) + '\n')
file.write("cto_g_30_SNR: " + str(cto_g_30_SNR) + '\n')
file.write("cto_s_05_SNR: " + str(cto_s_05_SNR) + '\n')
file.write("cto_s_10_SNR: " + str(cto_s_10_SNR) + '\n')