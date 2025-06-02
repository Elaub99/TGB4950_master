import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, distance_matrix
import pandas as pd

#------------------------------------------------------------------------------------#
#Back part of the pile    
las_file_back = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Verdal/Tur 17.03/Scan/all_scans_back.las")
points_back = np.vstack((las_file_back.x, las_file_back.y, las_file_back.z)).T

min_diameter_mm = 10
max_diameter_mm = 1900

eps_back = 0.012
min_samples_back = 7

dbscan_back = DBSCAN(eps_back, min_samples=min_samples_back)
labels_back = dbscan_back.fit_predict(points_back)
num_particles_back = len(np.unique(labels_back)) - 1
print(f"Number of particles identified in the back part of the pile: {num_particles_back}")

#------------------------------------------------------------------------------------#
#Front
las_file_front = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Verdal/Tur 17.03/Scan/all_scans_front.las")
points_front = np.vstack((las_file_front.x, las_file_front.y, las_file_front.z)).T

eps_front = 0.009
min_samples_front = 7

dbscan_front = DBSCAN(eps_front, min_samples=min_samples_front)
labels_front = dbscan_front.fit_predict(points_front)
num_particles_front = len(np.unique(labels_front)) - 1
print(f"Number of particles identified in the front part of the pile: {num_particles_front}")

#-----------------------------------------------------------------------------------#
# Defining Maximum Feret's diameter
def maximum_feret_diameter(points):
    n_points = len(points)
    if n_points == 1:
        return 0.0
    elif n_points == 2:
        return np.linalg.norm(points[0] - points[1])
    elif n_points == 3:
        return max(np.linalg.norm(points[0] - points[1]),
                   np.linalg.norm(points[0] - points[2]),
                   np.linalg.norm(points[1] - points[2]))
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        dist_matrix = distance_matrix(hull_points, hull_points)
        return np.max(dist_matrix)
    except:
        dist_matrix = distance_matrix(points, points)
        return np.max(dist_matrix)

#-----------------------------------------------------------------------------------#
results = []

# Process back part and add the data to results
for label in np.unique(labels_back):
    if label == -1:
        continue
    particle_points_back = points_back[labels_back == label]
    feret_mm = maximum_feret_diameter(particle_points_back) * 1000
    if feret_mm < min_diameter_mm or feret_mm > max_diameter_mm:
        continue
    results.append({
        "Particle ID": label,
        "Point Count": len(particle_points_back),
        "Feret Diameter (mm)": round(feret_mm, 3)
    })

# Process front part and add the data to results
for label in np.unique(labels_front):
    if label == -1:
        continue
    particle_points_front = points_front[labels_front == label]
    feret_mm = maximum_feret_diameter(particle_points_front) * 1000
    if feret_mm < min_diameter_mm or feret_mm > max_diameter_mm:
        continue
    label += num_particles_back
    results.append({
        "Particle ID": label,
        "Point Count": len(particle_points_front),
        "x [mm]": round(feret_mm, 3)
    })

# Sort results by descending Feret diameter values
results.sort(key=lambda r: r["Feret Diameter (mm)"], reverse=True)

    
# Add cumulative percentage based on number of points per particle
total_points = sum(r["Point Count"] for r in results)
cumulative = 0

for r in results:
    pct = r["Point Count"] / total_points
    cumulative += pct
    r["Pct of Total Points"] = round(pct * 100, 2)
    r["P(x) [%]"] = round((1- cumulative) * 100, 2)

# Find the largest particle
largest_diameter = round(max(r["Feret Diameter (mm)"] for r in results),2)

print('Predefined minimum size of all particles is: ', min_diameter_mm, 'mm \n'
      'Predefined maximum size of all particles is: ', max_diameter_mm, 'mm \n'
      'The largest particle found is: ', largest_diameter, 'mm \n'
      'Number of valid particles are ', len(results))

# Save results
df = pd.DataFrame(results)

# Save results
df = pd.DataFrame(results)
df.to_csv("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Koder/particle_sizes_pile_2.csv", index=False)
print("Particle sizes with Feret diameters and percentages saved to particle_sizes_pile_2.csv \n")

#--------------------------------------------------------------------------------------------#
# Visualisation
# Combining front and back part and normalising them
points = np.vstack((points_back, points_front))
global_min = np.min(points, axis=0)
points_back = points_back - global_min
points_front = points_front - global_min
points = np.vstack((points_back, points_front))


# Black and white model
# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
intensity_b = las_file_back.intensity
intensity_f = las_file_front.intensity
colors_b = np.vstack((intensity_b, intensity_b, intensity_b)).T / np.max(intensity_b)
colors_f = np.vstack((intensity_f, intensity_f, intensity_f)).T / np.max(intensity_f)
colors_bf = np.vstack((colors_b, colors_f))
point_cloud.colors = o3d.utility.Vector3dVector(colors_bf)

# Visualise the point cloud
o3d.visualization.draw_geometries([point_cloud])

# Give random colours to the particles
colors_back = np.zeros((len(points_back), 3))
for label in np.unique(labels_back):
    if label == -1:  # Noise (unclustered points)
        colors_back[labels_back == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_back = points_back[labels_back == label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_back)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Large particles in light grey
            colors_back[labels_back == label] = [0.8, 0.8, 0.8]  # Light grey (RGB 80%)
        elif feret_mm < min_diameter_mm:
            colors_back[labels_back == label] = [0.1, 0.1, 0.1]    
        else:  # Normal particles get random colors
            colors_back[labels_back == label] = np.random.rand(3)  # Random color

colors_front= np.zeros((len(points_front), 3))            
for label in np.unique(labels_front):
    if label == -1:  # Noise (unclustered points)
        colors_front[labels_front == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_front = points_front[labels_front== label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_front)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Large particles in light grey
            colors_front[labels_front == label] = [0.8, 0.8, 0.8]  # Light grey (RGB 80%)
        elif feret_mm < min_diameter_mm:
            colors_front[labels_front == label] = [0.1, 0.1, 0.1]    
        else:  # Normal particles get random colors
            colors_front[labels_front == label] = np.random.rand(3)  # Random color
            

colors = np.vstack((colors_back, colors_front))

# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualise the labeled point cloud
o3d.visualization.draw_geometries([point_cloud])

#--------------------------------------------------------------------------------------#

# Colour the particles based on their size
largest_diameter_25 = round(largest_diameter * 0.25,2)
largest_diameter_50 = round(largest_diameter * 0.50,2)
largest_diameter_75 = round(largest_diameter * 0.75,2)

print('The particle sizes and their colours are as follows: \n',
          'Black: invalid particle. Smaller than ', min_diameter_mm, 'mm \n',
          'Purple: larger than ', min_diameter_mm, 'mm, and smaller than ', largest_diameter_25, 'mm \n',
          'Orange: larger than ', largest_diameter_25, 'mm, and smaller than ', largest_diameter_50, 'mm \n',
          'Yellow: larger than ', largest_diameter_50, 'mm, and smaller than ', largest_diameter_75, 'mm \n',
          'Red: larger than ', largest_diameter_75, 'mm, and smaller than ', largest_diameter, 'mm \n',
          'Light grey: invalid particle. Larger than ', max_diameter_mm, 'mm')

# Colour back particles
colors_back = np.zeros((len(points_back), 3))
for label in np.unique(labels_back):
    if label == -1:  # Noise (unclustered points)
        colors_back[labels_back == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_back = points_back[labels_back == label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_back)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Invalid large particles
            colors_back[labels_back == label] = [0.8, 0.8, 0.8]  # Light grey
        elif feret_mm < min_diameter_mm: # Invalid small particles
            colors_back[labels_back == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: # 0-25% of the largest valid particle size
            colors_back[labels_back == label] = [0.5, 0.0, 0.5] # Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: # 25 - 50%
            colors_back[labels_back == label] = [1.0, 0.5, 0.0] # Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: #50 - 75%
            colors_back[labels_back == label] = [1.0, 1.0, 0.0] # Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: # 75 - 100%
            colors_back[labels_back == label] = [1.0, 0.0, 0.0] # Red


# Colour front particles
colors_front= np.zeros((len(points_front), 3))            
for label in np.unique(labels_front):
    if label == -1:  # Noise (unclustered points)
        colors_front[labels_front == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_front = points_front[labels_front== label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_front)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Invalid large particles
            colors_front[labels_front == label] = [0.8, 0.8, 0.8]  # Light grey
        elif feret_mm < min_diameter_mm: # Invalid small particles
            colors_front[labels_front == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: # 0 - 20% of the largest valid particle size
            colors_front[labels_front == label] = [0.5, 0.0, 0.5] # Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: # 20 - 40%
            colors_front[labels_front == label] = [1.0, 0.5, 0.0] # Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: # 40 - 60%
            colors_front[labels_front == label] = [1.0, 1.0, 0.0] # Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: # 60 - 80%
            colors_front[labels_front == label] = [1.0, 0.0, 0.0] # Red
            

colors = np.vstack((colors_back, colors_front))

# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the labeled point cloud
o3d.visualization.draw_geometries([point_cloud])