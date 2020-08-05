import numpy as np
import cv2

def masking(A, kernel):
    value = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            value += (A[i, j] * kernel[i, j])
    return value

def laplacian1(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap = np.zeros(im.shape, np.int)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            lap[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], kernel)
    return lap

def laplacian2(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]) / 3
    lap = np.zeros(im.shape, np.float32)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            lap[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], kernel)
    return lap

def laplacian_min(im):
    ext_im = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    kernel = np.array([[2, -1, 2], [-1, -4, -1], [2, -1, 2]]) / 3
    lap = np.zeros(im.shape, np.float32)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            lap[i-1][j-1] = masking(ext_im[i-1:i+2, j-1:j+2], kernel)
    return lap

def laplacian_gauss(im):
    ext_im = cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    kernel = np.array([[0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
                       [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                       [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                       [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                       [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                       [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
                       [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
                       [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
                       [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
                       [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
                       [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]])
    lap = np.zeros(im.shape, np.float32)
    for i in range(5, ext_im.shape[0] - 5):
        for j in range(5, ext_im.shape[1] - 5):
            lap[i-5][j-5] = masking(ext_im[i-5:i+6, j-5:j+6], kernel)
    return lap

def laplacian_dif(im):
    ext_im = cv2.copyMakeBorder(im, 5, 5, 5, 5, cv2.BORDER_REPLICATE)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    kernel = np.array([[-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
                       [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                       [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                       [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                       [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                       [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
                       [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
                       [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
                       [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
                       [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
                       [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]])
    lap = np.zeros(im.shape, np.float32)
    for i in range(5, ext_im.shape[0] - 5):
        for j in range(5, ext_im.shape[1] - 5):
            lap[i-5][j-5] = masking(ext_im[i-5:i+6, j-5:j+6], kernel)
    return lap

def threshold(G, threshold): 
    result = G
    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            if G[i][j] >= threshold:
                result[i][j] = 1
            elif G[i][j] <= -threshold:
                result[i][j] = -1
            else:
                result[i][j] = 0
    return result

def zerocross(result):
    ext_im = cv2.copyMakeBorder(result, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    ext_im = np.asarray(a=ext_im, dtype=np.int)
    cross = False
    ans = np.zeros(result.shape, np.int)
    for i in range(1, ext_im.shape[0] - 1):
        for j in range(1, ext_im.shape[1] - 1):
            cross = zerocross_helper(ext_im[i-1:i+2, j-1:j+2])
            if cross:
                ans[i-1][j-1] = 0
            else:
                ans[i-1][j-1] = 255
    return ans
           
def zerocross_helper(square):
    if square[1, 1] >= 1:
        for i in range(square.shape[0]):
            for j in range(square.shape[1]):
                if square[i, j] <= -1:
                    return True
        return False    
    else:
        return False
                      
im = cv2.imread('lena.bmp', 0)
laplacian_1 = zerocross(threshold(laplacian1(im), 15))
cv2.imwrite('laplacian_1.bmp', laplacian_1)
laplacian_2 = zerocross(threshold(laplacian2(im), 15))
cv2.imwrite('laplacian_2.bmp', laplacian_2)
laplacian_min = zerocross(threshold(laplacian_min(im), 20))
cv2.imwrite('laplacian_min.bmp', laplacian_min)
laplacian_gauss = zerocross(threshold(laplacian_gauss(im), 3000))
cv2.imwrite('laplacian_gauss.bmp', laplacian_gauss)
laplacian_dif = zerocross(threshold(laplacian_dif(im), 1))
cv2.imwrite('laplacian_dif.bmp', laplacian_dif)