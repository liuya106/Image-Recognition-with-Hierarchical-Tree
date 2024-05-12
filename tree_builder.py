import numpy as np
import cv2
import os
import matcher 

class Node:
    def __init__(self, parent=None, center=None):
        self.parent = parent
        self.center = center
        self.children = []
        self.descriptors = None
        self.leaf_idx = None
        
    def is_leaf(self):
        return len(self.children) == 0

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

def build_vocabulary_tree(descriptors, k, L):
    # Initialize root node with all descriptors
    root_node = Node()
    root_node.descriptors = descriptors

    # Recursively build tree with k-means clustering
    build_subtree(root_node, k, L)
    print(assign_leaf_indices(root_node))
    return root_node

def build_subtree(node, k, L):
    # Stop recursion if current node is a leaf or L layers have been reached
    if L == 0:
        return

    # Perform k-means clustering on node's descriptors
    try:
        _, labels, centers = cv2.kmeans(node.descriptors, k, None, 
                                        criteria=(cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01),
                                        attempts=5, flags=cv2.KMEANS_PP_CENTERS)
    except cv2.error as e:
        return 

    # Create child nodes for each cluster center and divide descriptors into clusters
    # print(f"parent descriptors is of shape {node.descriptors.shape}")
    for i in range(k):
        child_node = Node(parent=node, center=centers[i])
        child_descriptors = node.descriptors[labels.flatten() == i]
        #print(f"child descriptors is of shape {child_descriptors.shape}")

        if len(child_descriptors) > 0:
            child_node.descriptors = child_descriptors
            node.add_child(child_node)
            build_subtree(child_node, k, L-1)
    # print(f"Layer {2-L} has {len(node.children)} children")
    # print(f"children are {node.children}")

# we need this function to index our visual words in the tree leaves
def assign_leaf_indices(root):
    index = 0
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            node.leaf_idx = index
            index += 1
        else:
            stack.extend(node.children)
    return index

def compute_histogram(descriptors, root, k, L):
    num_descriptors = descriptors.shape[0]
    # Initialize histogram with zeros
    histogram = np.zeros(k**L, dtype=np.int32)

    # Traverse tree and update histogram
    for i, descriptor in enumerate(descriptors):
        node = root
        while not node.is_leaf():
            distances = [np.linalg.norm(child.center - descriptor) for child in node.children]
            node = node.children[np.argmin(distances)]
        histogram[node.leaf_idx] += 1

    return histogram


if __name__ == "__main__":
    path = './book_covers/'
    sift = cv2.SIFT_create()
    descriptors = []
    bag_of_words = []
    for filename in sorted(os.listdir(path), key=matcher.extract_integer):  # notice here the filename is sorted alphabatically, we need to sort them numerically
        img = cv2.imread(os.path.join(path, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        descriptors.append(des)
    descriptors_combined = np.concatenate(descriptors)
    root = build_vocabulary_tree(descriptors_combined, 10, 4)
    for img in descriptors:
        # print(compute_histogram(img, root, 10, 2).shape)
        bag_of_words.append(compute_histogram(img, root, 10, 4))
    bag_of_words = np.array(bag_of_words)

