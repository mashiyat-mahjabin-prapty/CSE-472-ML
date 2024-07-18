import cv2
import numpy as np
import matplotlib.pyplot as plt

def low_rank_approximation(mat, k):
    u, s, v = np.linalg.svd(mat)
    mat_approx = np.dot(u[:,:k], np.dot(np.diag(s[:k]), v[:k,:]))
    return mat_approx

# open the image
img = cv2.imread('image.jpg')

# convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print dimensions
print(gray.shape)

# reduce dimension to half
gray_half = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

# print dimensions
print(gray_half.shape)

# perform singular value decomposition
U, S, V = np.linalg.svd(gray_half)

# print(U, S, V)

rows, columns = gray_half.shape

plt.subplots(figsize=(10, 10))
plt.subplots_adjust(left=0.04, bottom=0.04, right=0.7, top=0.7, wspace=0.05, hspace=0.05)
i=1
j=1
# for each 100th rank, approximate the matrix and display all the images together
while i < min(rows, columns):
    mat_approx = low_rank_approximation(gray_half, i)
    plt.subplot(3, 4, j)
    plt.imshow(mat_approx, cmap='gray')
    plt.title('Rank ' + str(i))
    j = j+1
    if i >= 1 and i <= 50:
        i = i+10
    elif i > 50 and i <= 100:
        i = i+50
    elif i > 100 and i <= 400:
        i = i+100
    else:
        i = i+200

plt.suptitle('Image Reconstruction')
plt.tight_layout()
plt.show()

# k = 41, writer's name is visible
print('Writer\'s name is visible at rank 41')