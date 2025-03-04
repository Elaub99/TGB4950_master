# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:38:12 2025

@author: elaub
"""

import numpy as np
import open3d as o3d
import csv
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

def load_obj(file_path):
    mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=100000)  # Adjust sampling if needed
    return np.asarray(pcd.points)

def segment_rocks(points):
    clustering = DBSCAN(eps=0.05, min_samples=10).fit(points)
    labels = clustering.labels_
    unique_labels = set(labels)
    rock_clusters = {label: points[labels == label] for label in unique_labels if label != -1}
    return rock_clusters

def fit_ellipsoid(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    axes_lengths = 2 * np.sqrt(pca.explained_variance_)
    major_axis = max(axes_lengths)
    return major_axis

def process_rocks(file_path, output_csv):
    points = load_obj(file_path)
    rock_clusters = segment_rocks(points)
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Particle ID", "Particle Diameter"])
        
        for i, (rock_id, rock_points) in enumerate(rock_clusters.items()):
            diameter = fit_ellipsoid(rock_points)
            writer.writerow([i + 1, diameter])
    
    print(f"Processed {len(rock_clusters)} rocks. Data saved to {output_csv}")

# Example usage
process_rocks("rock_pile.obj", "rock_sizes.csv")
