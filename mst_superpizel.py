import numpy as np
from skimage.segmentation import felzenszwalb
from skimage import io
import networkx as nx
import scipy.io


def compute_average_rgb(image, superpixel_map):
    unique_labels = np.unique(superpixel_map)
    avg_rgb = {}
    
    for label in unique_labels:
        mask = superpixel_map == label
        avg_rgb[label] = np.mean(image[mask], axis=0)
    
    return avg_rgb

def l2_distance(feature1, feature2):
    return np.exp(np.linalg.norm(feature1 - feature2)/9)

def create_superpixel_graph_with_weights(superpixel_map, avg_rgb):
    G = nx.Graph()
    rows, cols = superpixel_map.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    for i in range(rows):
        for j in range(cols):
            current_label = superpixel_map[i, j]
            G.add_node(current_label)
            for dx, dy in neighbors:
                ni, nj = i + dx, j + dy
                if 0 <= ni < rows and 0 <= nj < cols:
                    neighbor_label = superpixel_map[ni, nj]
                    if neighbor_label != current_label:
                        weight = l2_distance(avg_rgb[current_label], avg_rgb[neighbor_label])
                        G.add_edge(current_label, neighbor_label, weight=weight)
    return G

def count_edges(graph):
    return graph.number_of_edges()

def compute_mst(graph):
    mst = nx.minimum_spanning_tree(graph, weight='weight')
    return mst




mat_data = scipy.io.loadmat('/home/cuonghoang/Downloads/mcg-2.0/pre-trained/demos/scg.mat')
image = io.imread('/home/cuonghoang/Downloads/mcg-2.0/pre-trained/demos/ILSVRC2012_val_00008229.JPEG')
# Apply Felzenszwalb superpixel segmentation to get a superpixel map
superpixel_map = mat_data['a2']

avg_rgb = compute_average_rgb(image, superpixel_map)
# Create the graph from the superpixel map
graph = create_superpixel_graph_with_weights(superpixel_map, avg_rgb)

# Count the number of edges in the graph
num_edges = count_edges(graph)
print(f"Number of edges in the graph: {num_edges}")

# Compute the Minimum Spanning Tree (MST)
mst = compute_mst(graph)
num_mst_edges = count_edges(mst)
print(f"Number of edges in the MST: {num_mst_edges}")

# Optionally, visualize the MST (requires matplotlib)
import matplotlib.pyplot as plt

pos = nx.spring_layout(mst)
nx.draw(mst, pos, with_labels=True, node_color='lightblue', edge_color='gray')
plt.title('Minimum Spanning Tree (MST)')
plt.show()
