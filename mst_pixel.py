import numpy as np
import cv2
import networkx as nx
import matplotlib.pyplot as plt

def create_pixel_graph(image):
    rows, cols = image.shape[:2]
    G = nx.Graph()

    # Add edges for 4-connectivity
    for i in range(rows):
        for j in range(cols):
            if j < cols - 1:  # Right neighbor
                weight = np.linalg.norm(image[i, j] - image[i, j + 1])
                G.add_edge((i, j), (i, j + 1), weight=weight)
            if i < rows - 1:  # Bottom neighbor
                weight = np.linalg.norm(image[i, j] - image[i + 1, j])
                G.add_edge((i, j), (i + 1, j), weight=weight)
            if j > 0:  # Left neighbor (optional, for symmetry)
                weight = np.linalg.norm(image[i, j] - image[i, j - 1])
                G.add_edge((i, j), (i, j - 1), weight=weight)
            if i > 0:  # Top neighbor (optional, for symmetry)
                weight = np.linalg.norm(image[i, j] - image[i - 1, j])
                G.add_edge((i, j), (i - 1, j), weight=weight)

    return G

def count_edges(graph):
    return graph.number_of_edges()

def compute_mst(graph):
    mst = nx.minimum_spanning_tree(graph, weight='weight')
    return mst

# Example usage
image_path = '/home/cuonghoang/Downloads/mcg-2.0/pre-trained/demos/ILSVRC2012_val_00008229.JPEG'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# If you want to use RGB image instead of grayscale
# image = cv2.imread(image_path)

# Create the pixel graph
graph = create_pixel_graph(image)

num_edges = count_edges(graph)
print(f"Number of edges in the graph: {num_edges}")
# Compute the Minimum Spanning Tree (MST)
mst = compute_mst(graph)
print(f"Number of edges in the MST: {mst.number_of_edges()}")

# Optionally, visualize the MST (simplified version for small images)
pos = {(i, j): (j, -i) for i in range(image.shape[0]) for j in range(image.shape[1])}
nx.draw(mst, pos=pos, node_size=1, width=0.1, with_labels=False)
plt.title('Minimum Spanning Tree (MST)')
plt.show()
