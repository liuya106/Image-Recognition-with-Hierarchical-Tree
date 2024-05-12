import cv2
import numpy as np
import os
import time
import re
import tree_builder


def extract_sift_features(img):
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, dp = sift.detectAndCompute(gray, None)
    return kp, dp

def build_vocabulary_tree(descriptors, k):
    # Define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    # Perform k-means clustering on descriptors
    _, labels, centers = cv2.kmeans(descriptors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return centers

def compute_histogram(descriptors, visual_words):
    # Compute the distance between each descriptor and each visual word
    #print(descriptors.shape, visual_words.shape)
    distances = np.linalg.norm(descriptors[:, np.newaxis, :] - visual_words, axis=2)
    # print(distances.shape)
    # Find the index of the closest visual word for each descriptor
    closest_words = np.argmin(distances, axis=1)
    # Compute the histogram of visual words
    histogram = np.bincount(closest_words, minlength=len(visual_words))
    return histogram

def extract_integer(str):
    integer = re.findall(r'\d+', str)
    if len(integer) == 0:
        print(f"There is no integer in {str}!!!")
    return int(integer[0])


def build_bag_of_words(path, vocab):
    sift = cv2.SIFT_create()
    descriptors = []
    bag_of_words = []
    for filename in sorted(os.listdir(path), key=extract_integer):  # notice here the filename is sorted alphabatically, we need to sort them numerically
        img = cv2.imread(os.path.join(path, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        descriptors.append(des)
    descriptors_combined = np.concatenate(descriptors)
    print(descriptors_combined.shape)

    # # this is 1 layer k-means clustering 
    # centers = build_vocabulary_tree(descriptors_combined, vocab)  # return was a 10*128 matrix
    # for img in descriptors:
    #     bag_of_words.append(compute_histogram(img, centers))
    # bag_of_words = np.array(bag_of_words)
    # return bag_of_words, centers

    # this is L layers hierarchical tree
    root = tree_builder.build_vocabulary_tree(descriptors_combined, 10, 4)
    for img in descriptors:
        bag_of_words.append(tree_builder.compute_histogram(img, root, 10, 4))
    bag_of_words = np.array(bag_of_words)
    return bag_of_words, root
    
    

def tfidf(bag_of_words):
    num_images, num_words = bag_of_words.shape
    #compute term frequency for each visual words in each image
    tf = bag_of_words / bag_of_words.sum(axis=1, keepdims=True)
    #compute inverse document frequency for each visual word in each image
    # +1 is to prevent zero division error
    idf = np.log(num_images / (1+ np.count_nonzero(bag_of_words, axis=0)))
    tfidf = tf*idf
    # Normalize the TF-IDF scores for each image
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf_norms = tfidf / norms
    return tfidf_norms

def build_inverted_index(bag_of_words):
    inverted_index = {}
    num_images = len(bag_of_words)

    for i in range(num_images):
        # Find the indices of the non-zero elements in the histogram
        word_indices = np.nonzero(bag_of_words[i])[0]

        # Update the inverted index for each non-zero element
        for j in word_indices:
            if j not in inverted_index:
                inverted_index[j] = [i]
            else:
                inverted_index[j].append(i)

    return inverted_index

def scoring(descriptors, database_dir):
    # vocab number determines the accuracy and time, a small vocab like 100 doesn't have enough accuracy
    start_time = time.time()
    bag_of_words, centers = build_bag_of_words(database_dir, 1000)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Build bag of words: {total_time} seconds")
    #print(bag_of_words)
    print(bag_of_words.shape)
    tfidf_bag = tfidf(bag_of_words)
    
    # query = compute_histogram(descriptors, centers)
    # switch to below when using vocabulary tree
    query = tree_builder.compute_histogram(descriptors, centers, 10, 4)
    norms = np.linalg.norm(query, keepdims=True)
    query_norms = query / norms

    inverted_index = build_inverted_index(bag_of_words)
    word_indices = np.nonzero(query)[0]
    images_found = []
    for word in word_indices:
        if word in inverted_index:
            images_found = list(set(images_found + inverted_index[word])) # make sure the elements are unique

    scores = {}
    for index in images_found:
        scores[np.dot(query_norms, tfidf_bag[index])] = index
    return dict(sorted(scores.items(), reverse=True))
        


if __name__ == "__main__":
    os.chdir("D:/CSC420/Final Project")
    start_time = time.time()
    print(os.getcwd())
    test_image = cv2.imread("./test_images/test1.png")
    keypoints, descriptors = extract_sift_features(test_image)
    print(descriptors.shape)
    book_cover_dir = "./book_covers"
    
    scores = scoring(descriptors, book_cover_dir)
    print(f"Top 10 retrievals are {list(scores.values())[:10]}")
    book_index = list(scores.values())[0] + 1
    best_match = cv2.imread(f"./book_covers/book{book_index}.jpg")
    cv2.imshow('The best match', best_match)
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time} seconds")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

    # # Load the two images
    # img1 = cv2.imread("./book_covers/book64.jpg", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread("./test_images/test4.jpg", cv2.IMREAD_GRAYSCALE)

    # # Initialize SIFT detector and descriptor
    # sift = cv2.SIFT_create()

    # # Find keypoints and descriptors for both images
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)

    # # Initialize brute-force matcher
    # bf = cv2.BFMatcher()

    # # Match the descriptors of the two images
    # matches = bf.knnMatch(des1, des2, k=2)

    # # Apply Lowe's ratio test to select good matches
    # good_matches = []
    # for m,n in matches:
    #     if m.distance < 0.75*n.distance:
    #         good_matches.append(m)
    
    # if len(good_matches) < 4: # not enough points to compute homography
    #     exit()

    # # Extract coordinates of keypoints in both images for good matches
    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    # # print(dst_pts.shape)

    # # Compute homography matrix using RANSAC algorithm
    # H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # # Apply homography to the keypoints of the first image to get their positions in the second image
    # h, w = img1.shape
    # #pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1,1,2)
    # sample_num = dst_pts.shape[0]
    # pts = np.zeros((sample_num, 1, 2), dtype=np.float32)
    # pts[:, 0, 0] = np.random.randint(0, w, size=(sample_num,))
    # pts[:, 0, 1] = np.random.randint(0, h, size=(sample_num,))
    # dst = cv2.perspectiveTransform(pts, H)
    # print(dst.shape)

    # # Compute the distance between the predicted positions and the actual positions of the keypoints in the second image
    # inliers = 0
    # for i in range(len(good_matches)):
    #     if mask[i]:
    #         dist = np.linalg.norm(dst_pts[i]-dst[i])
    #         if dist < 100:
    #             inliers += 1

    # # Print the number of inliers
    # print("Number of inliers:", inliers)

    # # Draw the matches and homography transformation on a new image 
    # img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None)
    # img_out = cv2.polylines(img2, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
    # cv2.imshow('Matches', img_matches)
    # cv2.imshow('Homography', img_out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()