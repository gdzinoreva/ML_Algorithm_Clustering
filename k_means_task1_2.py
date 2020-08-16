#This code implements the k-means clustering algorithm of Machine Learning


#import libraries
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import random


#defining function to calculate distance
def distance (point_a, point_b):
    dist = math.sqrt((point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2)
    return(dist)


#defining function to read data from csv file
def read_data(file_name):
    with open (file_name, 'r+') as f_csv:
        all_read = csv.reader(f_csv)
        x_y = {}
        i = 0
        for row in all_read:
            if i>0:
                x_y.update({row[0]: [float(row[1]), float(row[2])]})
            i += 1
    return (x_y)

    
#defining function to calculate mean
def d_mean(data_list):
    return np.mean(data_list, axis =0)


#defining function to calculate squared distance
def squaredist(point_a, point_b):
    dist = (point_a[0]-point_b[0])**2 + (point_a[1]-point_b[1])**2
    return(dist)


#defining function to calculate squared distance
def assignClusters(datalist, centerslist, distancelist, clusterslist):
    for entry in datalist:
        for i in range(0, len(centerslist)):
            dist = distance(centerslist[0], entry)
            distancelist.append(dist)
            distancelist[i] += distance(centerslist[i], entry)
            if min(distancelist) == distancelist[i]:
                 clusterslist.append(entry)   
        return(clusterslist)


#get list with x and y coordinates
all_content = read_data('data2008.csv')
x_ylist = list(all_content.values())
x_y_Array = np.array(x_ylist)


k = int(input("Enter k-value: "))
iterations = int(input("Enter number of iterations to perform: "))

clusters =  {i:[] for i in range(k)}
distances = {i:0 for i in range(k)}


centroids = random.sample(x_ylist, k)       #generate random centers to initialize cluster
print("\nThe initial centroids are: ", centroids)
print()


#loop over all data to assign to closest cluster
for iteration in range(0, iterations):
    for entry in x_ylist:
        
        distance_list = []    #list of distances from initial center
        for i in range(k):
            distances[i] = distance(centroids[i], entry)
            distance_list.append(distance(centroids[i], entry))
            
        #assign to closest center, update squared
        sq_dist = 0              #initialize mean square distance variable
        for a,v in distances.items():     
            if min(distance_list) == v :
                clusters[a].append(entry)
                sq_dist += squaredist(centroids[i], entry)
    
            
    #update mean
    for i,v in enumerate(centroids):
        centroids[i] = d_mean(clusters[i])
    print("Convergence, sum of square distance: ", sq_dist)


#countries belonging to each cluster   
countries = {i:[] for i in range(k)}

for j in all_content.items():
    for m,n in clusters.items():
        if j[1] in n:
            countries[m].append(j[0])        
print("\nCountries in their clusters: ", countries)


#number of countries per cluster
print("\nNumber of countries in each cluster respectively: ")
for a,v in countries.items():
    print(a, len(v))

print()
#mean of each cluster
for a,v in clusters.items():
    means1 = d_mean(v)
    print(f"Mean of Cluster{a}: {means1}")


###plot original data
##for item in x_ylist:
##    plt.scatter(x_y_Array[:, 0], x_y_Array[:, 1])
##plt.show()


#plot clustered data
for a, v in clusters.items():
    plt.scatter(np.array(clusters[a])[:, 0], np.array(clusters[a])[:, 1])

#show centers of each cluster
for a in  centroids:
    plt.scatter(a[0], a[1], marker = '+', s = 100)

plt.title('k-means Machine Learning Plot')
plt.xlabel('Birth Rate')
plt.ylabel('Life Expectancy')
plt.legend([i for i in range(k)])
plt.show()


