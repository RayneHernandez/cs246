#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:05:31 2018

@author: raynehernandez
"""

import re
import csv
import sys 
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial
from pyspark import SparkConf, SparkContext


def parseline(line):
    splitline = re.split(" ", line)
    splitline = [float(x) for x in splitline]
    return np.array(splitline)

def distanceNorm(data_point1, data_point2, distance):
    if distance == "euclidean":
        return spatial.distance.euclidean(data_point1, data_point2)
    elif distance == "manhattan":
        return spatial.distance.cityblock(data_point1, data_point2)

def assignCluster(data_point, centroid_list, distance):
    mindist = distanceNorm(data_point, centroid_list[0], distance)
    mindex = 0
    for i, centroid in enumerate(centroid_list):
        dist = distanceNorm(data_point, centroid, distance)
        if dist < mindist:
            mindist = dist
            mindex = i
    return (mindex, (mindist, data_point))

def getDistances(data_triple):
    return data_triple[1][0]

def tripleToTuple(data_triple):
    return (data_triple[0], data_triple[1][1])

def averageData(data_point1, data_point2):
    return (data_point1 + data_point2)/2

def getNewCentroids(data_tuple):
    return data_tuple[1]

def oneIteration(costs, data, centroid_list, distance):
    data_processed = data.map(lambda data_point: assignCluster(data_point, centroid_list, distance))
    distances = data_processed.map(getDistances)
    costs.append(distances.reduce(lambda x, y:x+y))
    data_clustered = data_processed.map(tripleToTuple)
    data_clustered_reduced = data_clustered.reduceByKey(averageData)
    new_centroids = data_clustered_reduced.map(getNewCentroids)
    centroid_list = new_centroids.collect()
    return centroid_list

def main():
    conf = SparkConf()
    sc = SparkContext(conf=conf)   
    
    lines = sc.textFile(sys.argv[1])
    centroid_lines = sc.textFile(sys.argv[2])
    
    

    
    distance_metrics = ["euclidean", "manhattan"]
    
    for metric in distance_metrics:
        costs = []
        data = lines.map(parseline)
        centroids = centroid_lines.map(parseline)
        centroid_list = centroids.collect()
        
        for i in range(20):
            centroid_list = oneIteration(costs, data, centroid_list, metric)
    
        plt.plot(costs)
        plt.xlabel("Iterations")
        plt.ylabel("Error")
        plt.title("Error over Iterations")
        plt.savefig("/Users/raynehernandez/Documents/CS_246/assign2/plot_{0}.png".format(metric))
        plt.close()
        
        with open("/Users/raynehernandez/Documents/CS_246/assign2/costs_{0}.csv".format(metric),  'w', newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerow(costs)
    
    sc.stop()

if __name__== "__main__":
    main()

