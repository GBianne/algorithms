# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 17:22:14 2025

@author: gbian
"""

import random
import matplotlib.pyplot as plt
import numpy as np

# grid size x,y
grid_size = (50, 50)

# grid resolution
grid_res = 0.01

# k input
k = 4

# points input
n_points = 500
points = []
for i in range(n_points):
    points.append((random.randint(0, grid_size[0]/grid_res)*grid_res, random.randint(0, grid_size[1]/grid_res)*grid_res))

# step 1 - generate initial k means
means = []
for i in range(k):
    means.append((random.randint(
        0, grid_size[0]/grid_res)*grid_res, random.randint(0, grid_size[1]/grid_res)*grid_res))

# plot initial state - hard coded
fig1, ax1 = plt.subplots()
ax1 = plt.scatter(*zip(*points), marker=".")
ax1 = plt.scatter(means[0][0], means[0][1], c='red', marker="s")
ax1 = plt.scatter(means[1][0], means[1][1], c='green', marker="s")
ax1 = plt.scatter(means[2][0], means[2][1], c='orange', marker="s")
ax1 = plt.scatter(means[3][0], means[3][1], c='black', marker="s")
plt.show()

# main loop - stop when convergence reached or when max iter reached
# convergence criteria: n_changes < cv_changes in point allegiance - default 95%-CV
#cv_changes = int(0.05*n_points)
cv_changes = 1
keep_going = True
max_iters = 100
iter_count = 0

# point allegiance assignment
assignments = [-1 for i in range(n_points)]

while keep_going and iter_count < max_iters:

    # number of point allegiance changes since last iter
    n_changes = 0

    # step 2 - for each point, calculate the closest mean
    for i in range(n_points):
        min_dist = -1
        closest_mean = -1
        for j in range(k):
            # compute the distance to the jth mean
            dist = np.sqrt((points[i][1]-means[j][1]) **
                           2 + (points[i][0]-means[j][0])**2)
            if min_dist == -1 or dist < min_dist:
                closest_mean = j
                min_dist = dist
        if closest_mean != assignments[i]:
            assignments[i] = closest_mean
            n_changes += 1

    # plot current state
    # if iter_count == 0:
    fig2, ax2 = plt.subplots()
    ax1 = plt.scatter(means[0][0], means[0][1], c='red', marker="s")
    ax1 = plt.scatter(means[1][0], means[1][1], c='green', marker="s")
    ax1 = plt.scatter(means[2][0], means[2][1], c='orange', marker="s")
    ax1 = plt.scatter(means[3][0], means[3][1], c='black', marker="s")
    for j in range(n_points):
        if assignments[j] == 0:
            ax2 = plt.scatter(points[j][0], points[j][1], c='red', marker=".")
        elif assignments[j] == 1:
            ax2 = plt.scatter(points[j][0], points[j]
                              [1], c='green', marker=".")
        elif assignments[j] == 2:
            ax2 = plt.scatter(points[j][0], points[j]
                              [1], c='orange', marker=".")
        else:
            ax2 = plt.scatter(points[j][0], points[j]
                              [1], c='black', marker=".")
    plt.show()

    # step 3 - reinitialise and calculate the new means
    means = [-1 for i in range(k)]
    counts = [0 for i in range(k)]
    for i in range(n_points):
        counts[assignments[i]] += 1
        if means[assignments[i]] == -1:
            # initialise
            means[assignments[i]] = points[i]
        else:
            # update the mean
            means[assignments[i]] = (
                means[assignments[i]][0] + points[i][0], means[assignments[i]][1] + points[i][1])
    # final update to the means
    for i in range(k):
        means[i] = (means[i][0]/counts[i], means[i][1]/counts[i])

    # end of iteration prints and check - are we done?!
    iter_count += 1
    print(f"This was iteration number {iter_count}, with {n_changes} allegiance changes!")
    if n_changes < cv_changes:
        # convergence reached!!
        keep_going = False







