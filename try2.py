import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate data from a standard normal distribution
mean = [0, 0]  # Mean of the distribution
cov = [[8, -5], [-1,-6]]  # Covariance matrix for a standard normal distribution

data = np.random.multivariate_normal(mean, cov, 1000)

# Step 2: Define a new covariance matrix for the desired shape
# For example, let's create an elliptical shape
# cov_new = [[4, 1.5], [1.5, 1]]  # This will stretch and rotate the data

# Apply the linear transformation
# L = np.linalg.cholesky(cov_new)  # Cholesky decomposition to get the transformation matrix
# transformed_data = data @ L.T

# Plot the original and transformed data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, edgecolor='k')
plt.title("Original Data (Circular)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(data[:, 0], data[:, 1], alpha=0.5, edgecolor='k')
plt.title("Transformed Data (Elliptical)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.axis('equal')

plt.tight_layout()
plt.show()
