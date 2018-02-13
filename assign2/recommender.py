#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 21:23:07 2018

@author: raynehernandez
"""

import re
import numpy as np

filename = "/Users/raynehernandez/Documents/CS_246/assign2/user-shows.txt"
movies = "/Users/raynehernandez/Documents/CS_246/assign2/shows.txt"


def readMatrix(filename):
    R = []
    with open(filename) as fp:
        for line in fp:
            row = re.split(" ", line)
            row = [int(x) for x in row]
            R.append(row)
    return np.array(R)

def readTitles(filename):
    titles = []
    with open(filename) as fp:
        for line in fp:
            titles.append(line)
    titles = [x[1:-2] for x in titles]
    return titles

def getQ(R):
    return np.diag(np.sum(R, axis=0))

def getP(R):
    return np.diag(np.sum(R, axis=1))

def negativeSqrt(M):
    return np.diag(1/np.diag(np.sqrt(M)))

def getUserUser(P, R, titles):
    P_nsqrt = negativeSqrt(P)
    gamma = ((P_nsqrt.dot(R)).dot((R.T).dot(P_nsqrt))).dot(R)
    recommendations = gamma[499,:100]
    top5 = sorted(range(len(recommendations)), key=lambda i: recommendations[i], reverse=True)[:5]
    shows = [titles[ind] for ind in top5]
    return shows

def getItemItem(Q, R, titles):
    Q_nsqrt = negativeSqrt(Q)
    gamma = R.dot((Q_nsqrt.dot(R.T)).dot((R).dot(Q_nsqrt)))
    recommendations = gamma[499, :100]
    top5 = sorted(range(len(recommendations)), key=lambda i: recommendations[i], reverse=True)[:5]
    shows = [titles[ind] for ind in top5]
    return shows

def main():
    titles = readTitles(movies)
    R = readMatrix(filename)
    Q = np.diag(np.sum(R, axis=0))
    P = np.diag(np.sum(R, axis=1))
    
    user_user_titles = getUserUser(P, R, titles)
    item_item_titles = getItemItem(Q, R, titles)
    
    print("User-User Titles")
    print(user_user_titles)
    print("Item-Item Titles")
    print(item_item_titles)
    
if __name__== "__main__":
  main()
