import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility


####optimal##########
# ### Generate first cluster with 1000 points centered around (5, 5)
# cluster1_x = np.random.normal(4, 1, 33353)
# cluster1_y = np.random.normal(5.5, 0.8, 33353)

# cluster1 = np.column_stack((cluster1_x, cluster1_y))

# theta = np.radians(30)  # Rotate by 30 degrees

# # Define the rotation matrix
# rotation_matrix = np.array([
#     [np.cos(theta), -np.sin(theta)],
#     [np.sin(theta), np.cos(theta)]
# ])

# # Apply the rotation matrix
# rotated_points = np.dot( cluster1,rotation_matrix)

# # print('rotated_points',rotated_points.shape)
# # exit()
# cluster1=rotated_points
# cluster1[:,0]=cluster1[:,0]-2
# cluster1[:,1]=cluster1[:,1]+2.5

# pos=np.where((((cluster1[:,0]-4)**2+(cluster1[:,1]-5)**2)<1))[0]
# print(pos.shape)
# # num_points_to_select = 3000

# pos1 = pos[0:1000]
# pos2 = pos[1001:2000]
# pos3 = pos[2001:3000]
# pos4 = pos[3001:4000]
# # print(pos1)
# # exit()
# cluster1[pos1,0]=cluster1[pos1,0]+1.1
# cluster1[pos1,1]=cluster1[pos1,1]-1.1

# cluster1[pos2,0]=cluster1[pos2,0]+0.3
# cluster1[pos2,1]=cluster1[pos2,1]-1.1


# cluster1[pos3,1]=cluster1[pos3,1]+1.7
# cluster1[pos3,0]=cluster1[pos3,0]-0.7


# cluster1[pos4,1]=cluster1[pos4,1]+2
# cluster1[pos4,0]=cluster1[pos4,0]

# # Generate second cluster with 50 points centered around (7, 7)
# cluster2_x = np.random.normal(7.8, 1, 439)
# cluster2_y = np.random.normal(7.5, 1.2, 439)
# cluster2 = np.column_stack((cluster2_x, cluster2_y))


# exit()

####small temp##########
# Generate first cluster with 1000 points centered around (5, 5)
# cluster1_x = np.random.normal(5.8, 2.5, 33353)
# cluster1_y = np.random.normal(6, 1.5, 33353)
# cluster1 = np.column_stack((cluster1_x, cluster1_y))

# # Generate second cluster with 50 points centered around (7, 7)
# cluster2_x = np.random.normal(7.7, 2, 439)
# cluster2_y = np.random.normal(6.5, 1.6, 439)
# cluster2 = np.column_stack((cluster2_x, cluster2_y))



# ######lard temp##########
# # Generate first cluster with 1000 points centered around (5, 5)
# cluster1_x = np.random.normal(6.4, 0.7, 33353)
# cluster1_y = np.random.normal(6, 0.6, 33353)
# cluster1 = np.column_stack((cluster1_x, cluster1_y))

# # Generate second cluster with 50 points centered around (7, 7)
# cluster2_x = np.random.normal(7, 0.4, 439)
# cluster2_y = np.random.normal(6.3, 0.5, 439)
# cluster2 = np.column_stack((cluster2_x, cluster2_y))



# ######ts temp##########
# # Generate first cluster with 1000 points centered around (5, 5)
cluster1_x = np.random.normal(5.4, 1.3, 33353)
cluster1_y = np.random.normal(6, 1, 33353)
cluster1 = np.column_stack((cluster1_x, cluster1_y))


theta = np.radians(-40)  # Rotate by 30 degrees

# Define the rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Apply the rotation matrix
rotated_points = np.dot( cluster1,rotation_matrix)
# print('rotated_points',rotated_points.shape)
# exit()
cluster1=rotated_points
cluster1[:,0]=cluster1[:,0]+5
cluster1[:,1]=cluster1[:,1]-2



pos=np.where((((cluster1[:,0]-5)**2+(cluster1[:,1]-6)**2)<1))[0]
print(pos.shape)
# num_points_to_select = 3000

pos1 = pos[0:1000]
pos2 = pos[1001:2000]
pos3 = pos[2001:3000]
pos4 = pos[3001:4000]


cluster1[pos1,0]=cluster1[pos1,0]+1.5
cluster1[pos1,1]=cluster1[pos1,1]-1.5

# cluster1[pos2,0]=cluster1[pos2,0]+0.3
cluster1[pos2,1]=cluster1[pos2,1]+2


cluster1[pos3,0]=cluster1[pos3,0]-0.5
cluster1[pos3,1]=cluster1[pos3,1]+1.5


cluster1[pos4,0]=cluster1[pos4,0]+0.5
cluster1[pos4,1]=cluster1[pos4,1]-2


# pos=np.where(cluster1[:,0]>7)[0]
# cluster1[pos,0]=cluster1[pos,0]-0.5


# Generate second cluster with 50 points centered around (7, 7)
cluster2_x = np.random.normal(8.4,1.1, 439)
cluster2_y = np.random.normal(6.3, 0.9, 439)
cluster2 = np.column_stack((cluster2_x, cluster2_y))



plt.figure(figsize=(12, 8))
# # Plot the clusters
plt.scatter(cluster1[:, 0], cluster1[:, 1], alpha=0.5, label='Cluster 1 (1000 points)', color='lime',s=1)
plt.scatter(cluster2[:, 0], cluster2[:, 1], label='Cluster 2 (50 points)', color='blue',s=10)
# plt.xlabel('Feature x')
# plt.ylabel('Feature y')
# plt.legend()
# plt.title('Generated Clusters')

plt.xlim(-1, 14)
plt.ylim(1, 12)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
# plt.savefig('optimal_temp.png')
# 
plt.show()

