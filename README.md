# k_nearest_neighbors_large_data
For the dataset Poker Hand, we use the training data consisting of approximately 25,000 examples (out of a domain of about 300,000,000) to classify a testing set using the k nearest neighbor classifier. Although the test set has 100,000 examples, we will likely find that it will take too long to test that many examples using Python. So it is acceptable to select, say, the first 10,000 examples. We write some code to read the datafile (which is comma delimited). We aldo treated the data in two different ways. First, we assume that the attributes are numerical. Second, we assume that the attributes are nominal. Final aim is to create a table that indicates the error rate (in percent) for our classification for 10 different values of k = {5, 25, 45, 65, 85, 105, 205, 305, 405, 505}.
