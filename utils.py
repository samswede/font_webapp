import numpy as np

# Retrieve array of top 8 recommendations from distance calculation

#calculate all distances and returns an unsorted list corresponding 
# with distances to each node with the corresponding index
def all_distances(point_1, all_points):
    #point_1 is tsne_results[index], all_points is tsne_results
    distances = [np.sqrt(np.sum((point_1-all_points[x])**2)) for x in range(len(all_points))]
    index = np.arange(len(all_points))  #...
    distances = np.stack((index, distances), axis=-1)

    return distances

#Sort shortest to longest
def sort_distances(array):
    #distances must be in 2nd column for this to work
    _sorted_array = array[np.argsort(array[:, 1])]
    return _sorted_array

#returns an array of the indices of recommended fonts from top to bottom
def return_index_recommendations(array_2d):
    recommendations = array_2d[:,0]
    recommendations = recommendations.astype(int)
    return recommendations

def create_recommendation_matrix(embedding):
  recommendation_matrix = np.zeros((len(embedding), len(embedding)))

  for i in range(len(embedding)):
    #compute distances from node i, and then sort by distance in ascending order
    sorted_distances = sort_distances(all_distances(embedding[i], embedding))
    #keep only sorted array of indices
    recommendation_indices = return_index_recommendations(sorted_distances)
    #create out edges in graph for node i
    recommendation_matrix[i] = recommendation_indices

      #creates 1s in the row (representing font node that needs recommendations) 
      #for all columns that correspond with the recommended fonts

  return recommendation_matrix.astype(int)
