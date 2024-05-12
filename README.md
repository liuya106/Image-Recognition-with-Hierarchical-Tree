# Image Recognition with Hierarchical Tree

<p align="center">**Introduction**</p>

This program provides a time-efficient solution for searching similar images from a massive
database via the vocabulary tree structure. The idea comes from Nister and Stewenius’
Paper Scalable Recognition with a Vocabulary Tree [1]. In this project, I took photos of a few books I have in hand and scraped 100 book cover images from Amazon.com. I built the book
cover matcher using both single-layer k-means clustering (10000 clusters) and vocabulary
tree method (4 layers and 10 branches each layer), did a query on them using the test
images, and measured their response time respectively. The time improvement, cut from
minutes to only seconds, is phenomenal. Additionally, it is again proven in this report that
more vocabulary increases search accuracy.
Implementation.


Introduction
This program provides a time-efficient solution for searching similar images from a massive
database via the vocabulary tree structure. The idea comes from Nister and Stewenius’
Paper Scalable Recognition with a Vocabulary Tree [1]. In this project, I took photos of a few
book I have in hand, and scraped 100 book cover images from Amazon.com. I built the book
cover matcher using both single layer k-means clustering (10000 clusters) and vocabulary
tree method (4 layers and 10 branches each layer), did a query on them using the test
images, and measured their response time respectively. The time improvement, cut from
minutes to only seconds, is phenomenal. Additionally, it is again proven in this report that
more vocabulary increases the search accuracy.
Implementation
● Web Crawler
The first Python script crawler.py performs web scraping on Amazon's website to download
book cover images with the keyword "sports". Because of Amazon’s bot detection
mechanism, I have to use GET requests with very detailed headers so that they could
bypass the detection [1], and then use BeautifulSoup to parse the HTML content of the
response. I then extract the book titles and cover image URLs from the search results using
BeautifulSoup's findall and select methods. The book cover images was downloaded to a
local folder, and metadata was saved to a CSV file using the pandas library. This prepare the
dataset of book cover images that we will be querying into later.
● Feature Extraction
Each book cover in the dataset is grayscaled first. Then, keypoints (contains coordinates,
scale, orientation) and descriptors (a n*128 matrix) are extracted using the SIFT
(Scale-Invariant Feature Transform) algorithm. I adopt Opencv’s implementation of this
algorithm [2], but you can also refer to my third assignment for the full implementation. Next,
we will need the help of keypoints and descriptors to train our clustering models and build
the vocabulary tree.
When I tried to concatenate descriptor matrices from different images, I used
sorted(os.listdir(path)) to returns a list of filenames. However, the list was sorted
alphabetically instead of numerically, like image1, image10, image100, image11, image12 …
, which completely messed up the order of the descriptors. This mistake led to some
completely incorrect search results and took me hours on debugging.
● Building the Vocabulary Tree
In tree_builder.py, I build a hierarchical vocabulary tree based on Nister and Stewenius'
paper [1]. The tree is built recursively with k-means clustering. The root node is initialized
with all descriptors, and then divided into k clusters. For each cluster, the descriptors closest
to the cluster are divided into k sub-clusters. This process is repeated until there are L
layers.
In terms of the tree structure, each node has a center, which is the mean of the descriptors
in that node. Each node also has a list of children, which are the nodes that resulted from
dividing the descriptors in that node. The tree is built recursively, with each node being a
parent to its children. The leaf nodes are the final clusters of descriptors. The attribute
leaf_idx is used to index leaves in the bag of words.
Notice that earlier in this program, only the single layer knn clustering (see
build_vocabulary_tree in matcher.py) was used to classify the descriptors, which was
extremely time-consuming. This provides a sharp contrast to the vocabulary tree approach
(see build_vocabulary_tree in tree_builder.py).
● Computing a Bag-of-Words Description
Now, we have established the structure of a vocabulary tree. We need to compute the bag of
words description for each image. The bag of words description is a 2D numpy array where
each row represents a histogram of visual words for an image. To compute the word
frequency in each image, I first computed the L2 normalized Euclidean distance between
descriptors and centers of the clusters, so the index of the closest visual word could be
found. I then count the number of occurrences for each word using np.bincount. Lastly, the
bag-of-words histograms are computed for both the query image and images in the
database.
● Inverted File Index
The build_inverted_index function takes in the bag_of_words matrix and returns an inverted
index dictionary. It relates the visual words to images in the database. If the visual word is
not already in the inverted index, it adds a new entry. This function offers a quick lookup to
exclude those who have none similarity with the query image.
● TF-IDF and Scoring
Prior to the scoring, TF-IDF (Term Frequency-Inverse Document Frequency) was used to
re-weigh the bag-of-words histogram. The goal is to strengthen the unique matches and to
weaken the universal matches among all images.
After applying TF-IDF, I obtained the scores for each image simply by computing the dot
product of the query image and the database images. Accordingly, the best 10 images were
retrieved from the database. It is shown that scoring with TF-IDF is an excellent indication of
similarity.
● Spatial Verification
Since two different images can still have many visual words in common, further verification
needs to be done on our matches using RANSAC with homography.
Firstly, a random set of coordinates was drawn from the image to estimate a homography
matrix. After estimating the homography matrix, I applied the homography transformation to
the keypoints of the first image to obtain their predicted positions in the second image. Then,
I computed the distance between the predicted positions and the actual positions of the
keypoints in the second image.
Finally, to determine the quality of matches, I counted the number of inliers from the best 10
images and determine the search results.
Time Efficiency of the Tree Structure
Theoretically, adopting the tree structure reduces the search time by a significant amount.
Imagine we are doing an image search on 1). the one-layer clustering with 1000 clusters. 2).
the three-layer vocabulary tree with 10 nodes on each layer. Hence, the number of
vocabulary stored in these two structures are the same, both 1000 words. If the query image
contains 100 descriptors, then we need 100*1000 computations for the one-layer clustering ,
whereas only 100*10*3 computations are needed for the three-layer tree. Therefore, the time
complexity is O(n*k**L) for the former and O(n*k*L) for the latter.
Now, let’s see how the tree structure performs in the actual program. The left image below is
a picture that I took yesterday, and the left one below has an index of 63 in our database.
Using OpenCV’2 SIFT implementation, 1248 descriptors were extracted from the left image.
Querying the image against the one-layer database with 1000 clusters takes 116 seconds,
whereas it only takes 17.888 seconds against the three-layer tree (10 nodes on each layer).
If we run against 10000 clusters, then it took nearly 20 minutes for the query to finish,
whereas it still takes less than half a minute for the tree.

  - Web Crawler
The first Python script crawler.py performs web scraping on Amazon's website to download
book cover images with the keyword "sports". Because of Amazon’s bot detection
mechanism, I have to use GET requests with very detailed headers so that they could
bypass the detection [1], and then use BeautifulSoup to parse the HTML content of the
response. I then extract the book titles and cover image URLs from the search results using
BeautifulSoup's findall and select methods. The book cover images was downloaded to a
local folder, and metadata was saved to a CSV file using the pandas library. This prepare the
dataset of book cover images that we will be querying into later.
- Feature Extraction
Each book cover in the dataset is grayscaled first. Then, keypoints (contains coordinates,
scale, orientation) and descriptors (a n*128 matrix) are extracted using the SIFT
(Scale-Invariant Feature Transform) algorithm. I adopt Opencv’s implementation of this
algorithm [2], but you can also refer to my third assignment for the full implementation. Next,
we will need the help of keypoints and descriptors to train our clustering models and build
the vocabulary tree.
When I tried to concatenate descriptor matrices from different images, I used
sorted(os.listdir(path)) to returns a list of filenames. However, the list was sorted
alphabetically instead of numerically, like image1, image10, image100, image11, image12 …
, which completely messed up the order of the descriptors. This mistake led to some
completely incorrect search results and took me hours on debugging.
● Building the Vocabulary Tree
In tree_builder.py, I build a hierarchical vocabulary tree based on Nister and Stewenius'
paper [1]. The tree is built recursively with k-means clustering. The root node is initialized
with all descriptors, and then divided into k clusters. For each cluster, the descriptors closest
to the cluster are divided into k sub-clusters. This process is repeated until there are L
layers.
In terms of the tree structure, each node has a center, which is the mean of the descriptors
in that node. Each node also has a list of children, which are the nodes that resulted from
dividing the descriptors in that node. The tree is built recursively, with each node being a
parent to its children. The leaf nodes are the final clusters of descriptors. The attribute
leaf_idx is used to index leaves in the bag of words.
Notice that earlier in this program, only the single layer knn clustering (see
build_vocabulary_tree in matcher.py) was used to classify the descriptors, which was
extremely time-consuming. This provides a sharp contrast to the vocabulary tree approach
(see build_vocabulary_tree in tree_builder.py).
● Computing a Bag-of-Words Description
Now, we have established the structure of a vocabulary tree. We need to compute the bag of
words description for each image. The bag of words description is a 2D numpy array where
each row represents a histogram of visual words for an image. To compute the word
frequency in each image, I first computed the L2 normalized Euclidean distance between
descriptors and centers of the clusters, so the index of the closest visual word could be
found. I then count the number of occurrences for each word using np.bincount. Lastly, the
bag-of-words histograms are computed for both the query image and images in the
database.
● Inverted File Index
The build_inverted_index function takes in the bag_of_words matrix and returns an inverted
index dictionary. It relates the visual words to images in the database. If the visual word is
not already in the inverted index, it adds a new entry. This function offers a quick lookup to
exclude those who have none similarity with the query image.
● TF-IDF and Scoring
Prior to the scoring, TF-IDF (Term Frequency-Inverse Document Frequency) was used to
re-weigh the bag-of-words histogram. The goal is to strengthen the unique matches and to
weaken the universal matches among all images.
After applying TF-IDF, I obtained the scores for each image simply by computing the dot
product of the query image and the database images. Accordingly, the best 10 images were
retrieved from the database. It is shown that scoring with TF-IDF is an excellent indication of
similarity.
● Spatial Verification
Since two different images can still have many visual words in common, further verification
needs to be done on our matches using RANSAC with homography.
Firstly, a random set of coordinates was drawn from the image to estimate a homography
matrix. After estimating the homography matrix, I applied the homography transformation to
the keypoints of the first image to obtain their predicted positions in the second image. Then,
I computed the distance between the predicted positions and the actual positions of the
keypoints in the second image.
Finally, to determine the quality of matches, I counted the number of inliers from the best 10
images and determine the search results.
Time Efficiency of the Tree Structure
Theoretically, adopting the tree structure reduces the search time by a significant amount.
Imagine we are doing an image search on 1). the one-layer clustering with 1000 clusters. 2).
the three-layer vocabulary tree with 10 nodes on each layer. Hence, the number of
vocabulary stored in these two structures are the same, both 1000 words. If the query image
contains 100 descriptors, then we need 100*1000 computations for the one-layer clustering ,
whereas only 100*10*3 computations are needed for the three-layer tree. Therefore, the time
complexity is O(n*k**L) for the former and O(n*k*L) for the latter.
Now, let’s see how the tree structure performs in the actual program. The left image below is
a picture that I took yesterday, and the left one below has an index of 63 in our database.
Using OpenCV’2 SIFT implementation, 1248 descriptors were extracted from the left image.
Querying the image against the one-layer database with 1000 clusters takes 116 seconds,
whereas it only takes 17.888 seconds against the three-layer tree (10 nodes on each layer).
If we run against 10000 clusters, then it took nearly 20 minutes for the query to finish,
whereas it still takes less than half a minute for the tree.
