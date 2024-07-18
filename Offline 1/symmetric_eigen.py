import numpy as np
import sys

# Take input for matrix dimension
n = int(input("Enter the dimension of the matrix: "))

while True:
    # Generate a random matrix of size 3x3
    random_matrix = np.random.randint(-2**30, (2**30)-1, size=(n, n))
    # check if determinant is zero
    if np.linalg.det(random_matrix) != 0:
        # make the matrix symmetric and the entries integer
        random_matrix = random_matrix + random_matrix.T
        break

print(random_matrix)

# perform eigenvalue decomposition
eigenvalues, eigenvectors = np.linalg.eig(random_matrix)

# print the eigenvalues and eigenvectors
print("Eigenvalues: ", eigenvalues)
print("Eigenvectors: ", eigenvectors)

# construct original matrix from eigenvectors
eigenvector_matrix = np.dot(eigenvectors, np.dot(np.diag(eigenvalues), np.linalg.inv(eigenvectors)))


# check if the product of the matrix and the eigenvector matrix is equal to the product of the eigenvalue and the eigenvector
for i in range(n):
    if np.allclose(random_matrix, eigenvector_matrix):
        print("True")
    else:
        print("False")
        sys.exit()

print("Eigenvalue decomposition is correct")