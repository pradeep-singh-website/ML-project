def knn(data,query,k,distance_fn,choice_fn):
  neighbor_distances_and_indices = []

  # For each example in the data
  for index, example in enumerate(data):
    
    # calculate the distance between the query example and the current
    # example from the data
    distance = distance_fn(example[:-1],query)

    neighbor_distances_and_indices.append((distance,index))

  sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
  k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

  k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

  return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)

def mean(labels):
  return sum(labels) / len(labels)

from collections import Counter
def mode(labels):
  return Counter(labels).most_common(1)[0][0]

import math
def euclidean_distance(point1,point2):
  sum_sq_distance = 0
  for i in range(len(point1)):
    sum_sq_distance += math.pow(point1[i]-point2[i],2)
  return math.sqrt(sum_sq_distance)

from sklearn.datasets import load_iris
from pprint import pprint

iris = load_iris()

x = iris.data
y = iris.target

clf_data = []

for i in range(len(y)):
  clf_data.append((*x[i],y[i]))

clf_data


clf_query = [4.1, 2.1, 3.9, 1.8]
clf_k_nearest_neighbors, clf_prediction = knn(
    clf_data, clf_query, k=5, distance_fn=euclidean_distance, choice_fn=mode
)
print(clf_prediction,clf_k_nearest_neighbors)