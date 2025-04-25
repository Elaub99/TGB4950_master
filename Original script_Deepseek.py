# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 10:48:47 2025

@author: elaub
"""
#Step 1 - Read the .las file and cluster the particles
import laspy
import numpy as np
from sklearn.cluster import DBSCAN

# Open the .las file
las_file = laspy.read("your_file.las")

# Extract x, y, z coordinates
points = np.vstack((las_file.x, las_file.y, las_file.z)).T

# DBSCAN clustering
dbscan = DBSCAN(eps=0.05, min_samples=10)  # Adjust parameters as needed
labels = dbscan.fit_predict(points)

# Number of unique particles (excluding noise, labeled as -1)
num_particles = len(np.unique(labels)) - 1
print(f"Number of particles identified: {num_particles}")

#Step 2 - Fit ellipsoid and calculate diameter/Volume
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
import pandas as pd

# Function to fit an ellipsoid to a set of points
def fit_ellipsoid(points):
    def residuals(params, points):
        a, b, c = params
        return np.sum((points[:, 0]**2 / a**2 + points[:, 1]**2 / b**2 + points[:, 2]**2 / c**2 - 1)**2)

    # Initial guess for ellipsoid axes (a, b, c)
    initial_guess = [1.0, 1.0, 1.0]
    result = least_squares(residuals, initial_guess, args=(points,))
    return result.x

# Function to calculate the diameter of the enclosing square
def calculate_diameter(ellipse_axes):
    # The diameter is the side length of the square enclosing the ellipse
    return 2 * max(ellipse_axes)

# Function to calculate the volume of the ellipsoid
def calculate_volume(ellipsoid_axes):
    a, b, c = ellipsoid_axes
    return (4/3) * np.pi * a * b * c

# Process each particle
results = []
for label in np.unique(labels):
    if label == -1:  # Skip noise
        continue

    # Get points for the current particle
    particle_points = points[labels == label]

    # Fit ellipsoid
    ellipsoid_axes = fit_ellipsoid(particle_points)

    # Calculate diameter (using intermediate and smaller axes)
    ellipse_axes = sorted(ellipsoid_axes)[1:]  # Intermediate and smaller axes
    diameter = calculate_diameter(ellipse_axes)

    # Calculate volume
    volume = calculate_volume(ellipsoid_axes)

    # Save results
    results.append({
        "Particle ID": label,
        "Diameter": diameter,
        "Volume": volume
    })

# Save results to a CSV file
df = pd.DataFrame(results)
df.to_csv("particle_sizes.csv", index=False)
print("Particle sizes saved to particle_sizes.csv")

#Step 3 - Visualize one particle with diameter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Select a particle to visualize (e.g., the first particle)
particle_id = results[0]["Particle ID"]
particle_points = points[labels == particle_id]

# Fit ellipsoid to the selected particle
ellipsoid_axes = fit_ellipsoid(particle_points)

# Calculate diameter line endpoints
diameter_length = results[0]["Diameter"]
midpoint = np.mean(particle_points, axis=0)
diameter_line = np.array([
    midpoint - [diameter_length / 2, 0, 0],
    midpoint + [diameter_length / 2, 0, 0]
])

# Plot the particle and diameter line
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(particle_points[:, 0], particle_points[:, 1], particle_points[:, 2], s=1, c='b', label="Particle")
ax.plot(diameter_line[:, 0], diameter_line[:, 1], diameter_line[:, 2], c='r', linewidth=2, label="Diameter")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()