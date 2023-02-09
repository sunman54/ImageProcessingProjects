import cv2 as cv
import numpy as np,sys
from PIL import Image


A = cv.imread('./img/summer.jpg')
B = cv.imread('./img/winter.jpg')

rang = 4

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(rang):
    G = cv.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(rang):
    G = cv.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A

lpA = [gpA[4]]
for i in range(rang,0,-1):
    size = (gpA[i - 1].shape[1], gpA[i - 1].shape[0])
    GE = cv.pyrUp(gpA[i],dstsize=size)
    L = cv.subtract(gpA[i-1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[4]]
for i in range(rang,0,-1):
    size = (gpB[i - 1].shape[1], gpB[i - 1].shape[0])
    GE = cv.pyrUp(gpB[i],dstsize=size)
    L = cv.subtract(gpB[i-1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level

LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    LS.append(ls)
    print(la.shape)
    print(lb.shape)

# now reconstruct
ls_ = LS[0]
for i in range(1, len(LS)):
    ls_ = cv.pyrUp(ls_)
    print(ls_.shape, LS[i].shape)
    ls_ = cv.add(ls_, LS[i])

cv.imwrite('result.jpg',ls_)




real = np.hstack((A[:,:cols//2],B[:,cols//2:]))
cv.imwrite('Direct_blending_black_white.jpg',real)


