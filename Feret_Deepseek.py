import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, distance_matrix
import pandas as pd


# Open the .las file
las_file = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Verdal/Tur 17.03/Scan/1001.las")

# Extract x, y, z coordinates and normalize
points = np.vstack((las_file.x, las_file.y, las_file.z)).T
points -= np.min(points, axis=0)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.011, min_samples=10)
labels = dbscan.fit_predict(points)
num_particles = len(np.unique(labels)) - 1
print(f"Number of particles identified: {num_particles}")


def maximum_feret_diameter(points):
    """
    Compute the maximum Feret diameter of a 3D point cloud with error handling.
    For small clusters, returns the maximum pairwise distance.
    """
    n_points = len(points)

    # Handle degenerate cases
    if n_points == 1:
        return 0.0  # Single point has no diameter
    elif n_points == 2:
        return np.linalg.norm(points[0] - points[1])  # Simple distance for 2 points
    elif n_points == 3:
        # For 3 points, return maximum pairwise distance
        return max(np.linalg.norm(points[0] - points[1]),
                   np.linalg.norm(points[0] - points[2]),
                   np.linalg.norm(points[1] - points[2]))

    try:
        # Attempt convex hull for 4+ points
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # Calculate all pairwise distances
        dist_matrix = distance_matrix(hull_points, hull_points)
        return np.max(dist_matrix)
    except:
        # Fallback for coplanar points or other hull failures
        dist_matrix = distance_matrix(points, points)
        return np.max(dist_matrix)


# Modify your particle processing loop to filter small clusters:
min_particle_points = 4  # Minimum points to consider as a valid particle

results = []
for label in np.unique(labels):
    if label == -1:  # Skip noise
        continue

    particle_points = points[labels == label]

    if len(particle_points) < min_particle_points:
        print(f"Skipping small cluster {label} with {len(particle_points)} points")
        continue

    feret_diameter = maximum_feret_diameter(particle_points)

    # Volume calculation with error handling
    try:
        hull = ConvexHull(particle_points)
        volume = hull.volume
    except:
        volume = 0  # Or np.nan if you prefer

    results.append({
        "Particle ID": label,
        "Point Count": len(particle_points),
        "Feret Diameter (m)": round(feret_diameter,5),
        "Feret Diameter (mm)": round(feret_diameter,3) * 1000,
        "Volume (m³)": round(volume,5),
        "Volume (mm³)": volume * 1e9
    })
    results.sort(key=lambda result: result["Feret Diameter (mm)"])

# Save results
df = pd.DataFrame(results)
df.to_csv("particle_sizes_feret.csv", index=False)
print("Particle sizes with Feret diameters saved to particle_sizes_feret.csv")

# Visualisation
# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Color points based on their cluster label
colors = np.zeros((len(points), 3))
for label in np.unique(labels):
    if label == -1:  # Noise (unclustered points)
        colors[labels == label] = [0, 0, 0]  # Black for noise
    else:
        colors[labels == label] = np.random.rand(3)  # Random color for each particle

point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualise the labeled point cloud
o3d.visualization.draw_geometries([point_cloud])

