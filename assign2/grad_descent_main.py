#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:03:37 2018

@author: raynehernandez
"""

import re
import numpy as np
import matplotlib.pyplot as plt

## have access file with python indexing             
def getRating(filename, movie_id, user_id):
    with open(filename) as fp:
        for line in fp:
            data = re.split("\t", line)
            data = [int(x) for x in data]
            user = data[0]
            movie = data[1]
            if (movie - 1) == movie_id and (user - 1) == user_id:
                return data[2]
        return 0          
            
def maxUserAndMovie(filename):
    max_user = 0
    max_movie = 0
    with open(filename) as fp:
        for line in fp:
            data = re.split("\t", line)
            data = [int(x) for x in data]
            user = data[0]
            movie = data[1]
            if user > max_user:
                max_user = user
            if movie > max_movie:
                max_movie = movie
        fp.close()
        return max_user, max_movie
        

def gradQ(Q, P, i, j, filename, n):
    summ = 0
    for u in range(n):
        summ += (getRating(filename, i, u) - Q[i,].dot(P[u,].T))*P[u,j]
    return summ

def updateQEntry(Q, P, i, j, lam, eta, filename, n):
    ret = Q[i, j] - eta*(2*lam*Q[i, j] + 2*gradQ(Q, P, i, j, filename, n))
    return ret

def gradP(Q, P, u, j, filename, m):
    summ = 0
    for i in range(m):
        summ += (getRating(filename, i, u) - Q[i,].dot(P[u,].T))*Q[i, j]
    return summ

def updatePEntry(Q, P, u, j, lam, eta, filename, m):
    ret = P[u, j] - eta*(2*lam*P[u, j] - 2*gradP(Q, P, u, j, filename, m))
    return ret

def lossE(Q, P, lam, filename, m, n):
    summ = 0
    for i in range(m):
        for u in range(n):
            summ += (getRating(filename, i, u ) - Q[i,].dot(P[u,].T))**2
    
    for u in range(n):
        summ += lam*(np.linalg.norm(P[u,]))**2
    
    for i in range(m):
        summ += lam*(np.linalg.norm(Q[i,]))**2
    
    return summ


def main():
    
    #parameters 
    filename = "/Users/raynehernandez/Documents/CS_246/assign2/ratings.train.txt"
    
    k = 20 
    lam = 0.1
    eta = 0.001
    
    
    #get dimensions
    n, m = maxUserAndMovie(filename)
    
    #initialize matrices
    Q = np.random.rand(m,k)*np.sqrt(5/k)
    P = np.random.rand(n,k)*np.sqrt(5/k)
    
    #lossE(Q, P, lam, filename, m, n)
    errors = [lossE(Q, P, lam, filename, m, n)]
    
    for iteration in range(40):
        
        ##run one update Q
        for i in range(m):
            for j in range(k):
                Q[i, j] = updateQEntry(Q, P, i, j, lam, eta, filename, n)
        
        #run one update P
        for u in range(n):
            for j in range(k):
                P[u, j] = updatePEntry(Q, P, u, j, lam, eta, filename, m)
                    
            errors.append(lossE(Q, P, lam, filename, m, n))
    
    plt.plot(errors)
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.title('Error over each Iteration (eta = {})'.format(eta))
    plt.savefig("/Users/raynehernandez/Documents/CS_246/assign2/recommendation_plot.png")

        
if __name__== "__main__":
  main()









