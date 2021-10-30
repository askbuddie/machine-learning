# Author: Nishant Banjade


# Euclidian Distance 

# X1 = [x1, x2, x3, x4 ....]

# Y1 = [y1, y2, y3,.....]

# Distance = sum( y1 - x1 ) ^2 
from math import sqrt
def Euclidian_distance(X1, Y1):
    sum = 0
    for i in range(len(X1)-1):
        sum+=(X1[i] - Y1[i])**2
        return sqrt(sum)




# Now lets get the neighbours that are nearer

def get_neighbors(data, test_data, k):
    # k is the number of neighbours

    distances = list()
    for temp in data:
        dist = Euclidian_distance(test_data, temp)

        distances.append((temp, dist))
        # sort the list 
    distances.sort(key = lambda tup: tup[1])
    neighbours = list()

    # after sorting find top 3 value
    for i in range(k):
        neighbours.append(distances[i][0])
    return neighbours


# Prediction part 
def prediction(data, test_data, k):
    final_neighbour = get_neighbors(data, test_data, k)
    # prediction index at the last column 
    get_output = [r[-1] for r in final_neighbour]

    predict = max(set(get_output), key = get_output.count)
    return predict
# test the code 
dataset = [[4.7810836,2.550537003,0],
	[9.465489372,2.362125076,0],
	[4.396561688,4.400293529,0],
	[2.38807019,1.850220317,0],
	[2.06407232,3.005305973,0],
	[5.627531214,2.759262235,1],
	[1.332441248,2.088626775,1],
	[4.922596716,1.77106367,1],
	[2.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]


pred = prediction(dataset, dataset[0], 3)

print("Expected %d , Predicted %d " % (dataset[0][-1], pred))
