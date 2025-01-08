import scipy.io
from skimage import io, segmentation, color
# Load the .mat file
import numpy as np

mat_data = scipy.io.loadmat('/home/cuonghoang/Downloads/mcg-2.0/pre-trained/demos/mcg.mat')
image = io.imread('/home/cuonghoang/Downloads/mcg-2.0/pre-trained/demos/ILSVRC2012_val_00008229.JPEG')
# Access the data
# print(mat_data)
print(np.unique(mat_data['a1']).shape)
# boundaries_image = segmentation.mark_boundaries(image, mat_data['a2'],color=(1, 1, 1))
# io.imsave('mcg.jpg', (boundaries_image * 255).astype(np.uint8))
print('image',image)

def count_edges(superpixel_map):
    # Get the unique superpixel labels
    unique_labels = np.unique(superpixel_map)
    
    # Initialize a set to keep track of unique edges
    edges = set()
    
    # Define the 8-connectivity neighborhood
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Get the dimensions of the superpixel map
    rows, cols = superpixel_map.shape
    
    for i in range(rows):
        for j in range(cols):
            current_label = superpixel_map[i, j]
            for dx, dy in neighbors:
                ni, nj = i + dx, j + dy
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_label = superpixel_map[ni, nj]
                    if neighbor_label != current_label:
                        # Add the edge as an ordered tuple to avoid duplicates
                        edge = tuple(sorted((current_label, neighbor_label)))
                        edges.add(edge)
    
    # The number of unique edges
    return len(edges)

print(count_edges(mat_data['a1']))