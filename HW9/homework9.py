import numpy as np
import cv2
import math

def masking(A, kernel):
    value = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            value += (A[i, j] * kernel[i, j])
    return value

def roberts(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    k1 = np.array([[-1, 0], [0, 1]])
    k2 = np.array([[0, -1], [1, 0]])
    Gx = np.zeros(im.shape, np.int)
    Gy = np.zeros(im.shape, np.int)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            Gx[i-1][j-1] = masking(ext_im[i:i+2, j:j+2], k1)
            Gy[i-1][j-1] = masking(ext_im[i:i+2, j:j+2], k2)
    roberts_G = np.zeros((im.shape[0], im.shape[1]), np.int)
    roberts_G = np.sqrt(Gx ** 2 + Gy ** 2)
    return roberts_G

def threshold(G, threshold):
    result = G
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i][j] >= threshold:
                result[i][j] = 0
            else:
                result[i][j] = 255
    return result

def prewitts(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    k1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    k2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    Gx = np.zeros(im.shape, np.int)
    Gy = np.zeros(im.shape, np.int)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            Gx[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], k1)
            Gy[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], k2)
    prewitts_G = np.zeros((im.shape[0], im.shape[1]), np.int)
    prewitts_G = np.sqrt(Gx ** 2 + Gy ** 2)
    return prewitts_G        
          
def sobel(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    k1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    k2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gx = np.zeros(im.shape, np.int)
    Gy = np.zeros(im.shape, np.int)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            Gx[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], k1)
            Gy[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], k2)
    sobel_G = np.zeros((im.shape[0], im.shape[1]), np.int)
    sobel_G = np.sqrt(Gx ** 2 + Gy ** 2)
    return sobel_G     
      
def frei_and_chen(im):###############################################################
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    k1 = np.array([[-1, -math.sqrt(2), -1], [0, 0, 0], [1, math.sqrt(2), 1]])
    k2 = np.array([[-1, 0, 1], [-math.sqrt(2), 0, math.sqrt(2)], [-1, 0, 1]])
    Gx = np.zeros(im.shape, np.float32)
    Gy = np.zeros(im.shape, np.float32)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            Gx[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], k1)
            Gy[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], k2)
    fac_G = np.zeros((im.shape[0], im.shape[1]), np.int)
    fac_G = np.sqrt(Gx ** 2 + Gy ** 2)
    return fac_G

def kirsch(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    k1 = np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]])
    k2 = np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
    k3 = np.array([[5, 5, 5], [-3, 0 ,-3], [-3, -3, -3]])
    k4 = np.array([[5, 5, -3], [5, 0 , -3], [-3, -3, -3]])
    k5 = np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]])
    k6 = np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]])
    k7 = np.array([[-3, -3, -3], [-3, 0 ,-3], [5, 5, 5]])
    k8 = np.array([[-3, -3, -3], [-3, 0 , 5], [-3, 5, 5]])
    kirsch_G = np.zeros(im.shape, np.int)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            for kernel in [k1, k2, k3, k4, k5, k6, k7, k8]:
                x = masking(ext_im[i-1:i+2, j-1:j+2], kernel)
                if x > kirsch_G[i-1, j-1]:
                    kirsch_G[i-1][j-1] = x
    return kirsch_G

def robinson(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    k1 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    k2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
    k3 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    k4 = np.array([[2, 1, 0], [1, 0 , -1], [0, -1, -2]])
    k5 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    k6 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
    k7 = np.array([[-1, -2, -1], [0, 0 ,0], [1, 2, 1]])
    k8 = np.array([[-2, -1, 0], [-1, 0 , 1], [0, 1, 2]])
    robinson_G = np.zeros(im.shape, np.int)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            for kernel in [k1, k2, k3, k4, k5, k6, k7, k8]:
                x = masking(ext_im[i-1:i+2, j-1:j+2], kernel)
                if x > robinson_G[i-1, j-1]:
                    robinson_G[i-1][j-1] = x
    return robinson_G

def babu(im):
    ext_im = cv2.copyMakeBorder(im, 2, 2, 2, 2, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    k1 = np.array([[100, 100, 100, 100, 100], [100, 100, 100, 100, 100], [0, 0, 0, 0, 0], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]])
    k2 = np.array([[100, 100, 100, 100, 100], [100, 100, 100, 78, -32], [100, 92, 0, -92, -100], [32, -78, -100, -100, -100], [-100, -100, -100, -100, -100]])
    k3 = np.array([[100, 100, 100, 32, -100], [100, 100, 92, -78, -100], [100, 100, 0, -100, -100], [100, 78, -92, -100, -100], [100, -32, -100, -100, -100]])
    k4 = np.array([[-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, 0, 100, 100]])
    k5 = np.array([[-100, 32, 100, 100, 100], [-100, -78, 92, 100, 100], [-100, -100, 0, 100, 100], [-100, -100, -92, 78, 100], [-100, -100, -100, -32, 100]])
    k6 = np.array([[100, 100, 100, 100, 100], [-32, 78, 100, 100, 100], [-100, -92, 0, 92, 100], [-100, -100, -100, -78, 32], [-100, -100, -100, -100, -100]])
    babu_G = np.zeros(im.shape, np.int)
    for i in range(2, ext_im.shape[0] - 2):
        for j in range(2, ext_im.shape[1] - 2):
            for kernel in [k1, k2, k3, k4, k5, k6]:
                x = masking(ext_im[i-2:i+3, j-2:j+3], kernel)
                if x > babu_G[i-2, j-2]:
                    babu_G[i-2][j-2] = x
    return babu_G

im = cv2.imread('lena.bmp', 0)
roberts_G = threshold(roberts(im), 12)
cv2.imwrite('roberts_G.bmp', roberts_G)
prewitts_G = threshold(prewitts(im), 24)
cv2.imwrite('prewitts_G.bmp', prewitts_G)
sobel_G = threshold(sobel(im), 38)
cv2.imwrite('sobel_G.bmp', sobel_G)
fac_G = threshold(frei_and_chen(im), 30)
cv2.imwrite('fac_G.bmp', fac_G)
kirsch_G = threshold(kirsch(im), 135)
cv2.imwrite('kirsch_G.bmp', kirsch_G)
robinson_G = threshold(robinson(im), 43)
cv2.imwrite('robinson_G.bmp', robinson_G)
babu_G = threshold(babu(im), 12500)
cv2.imwrite('babu_G.bmp', babu_G)