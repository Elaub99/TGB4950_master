import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, distance_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
#------------------------------------------------------------------------------------#
#Back part of the pile
# Open the .las file
las_file_back = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Ballangen/Scan/Pile 1_back.las")
points_back = np.vstack((las_file_back.x, las_file_back.y, las_file_back.z)).T

min_diameter_mm = 50
max_diameter_mm = 1500

eps_back = 0.015
min_samples_back = 7

dbscan_back = DBSCAN(eps_back, min_samples=min_samples_back)
labels_back = dbscan_back.fit_predict(points_back)
num_particles_back = len(np.unique(labels_back)) - 1
print(f"Number of particles identified in the back part of the pile: {num_particles_back}")

#------------------------------------------------------------------------------------#
#Front
las_file_front = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Ballangen/Scan/Pile 1_front.las")
points_front = np.vstack((las_file_front.x, las_file_front.y, las_file_front.z)).T

eps_front = 0.0077
min_samples_front = 7

dbscan_front = DBSCAN(eps_front, min_samples=min_samples_front)
labels_front = dbscan_front.fit_predict(points_front)
num_particles_front = len(np.unique(labels_front)) - 1
print(f"Number of particles identified in the front part of the pile: {num_particles_front}")

#------------------------------------------------------------------------------------#
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

# Process back part
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

# Process front part
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
        "Feret Diameter (mm)": round(feret_mm, 3)
    })

# Sort results by Feret diameter descending
results.sort(key=lambda r: r["Feret Diameter (mm)"], reverse=True)

# Add normalized diameter and cumulative percentage
total_diameter = sum(r["Feret Diameter (mm)"] for r in results)
cumulative = 0
for r in results:
    pct = r["Feret Diameter (mm)"] / total_diameter
    r["Pct of Total Diameter"] = pct
    r["Cumulative %"] = (1 - cumulative) * 100
    cumulative += pct

largest_diameter = round(max(r["Feret Diameter (mm)"] for r in results),2)

print('Min size of the entire pile of particles is: ', min_diameter_mm, 'mm \n'
      'Max size of particles is: ', max_diameter_mm, 'mm \n'
      'The largest particle found is: ', largest_diameter, 'mm \n'
      'Number of valid particles are ', len(results))

# Save results
df = pd.DataFrame(results)
df.to_csv("particle_sizes_pile_1.csv", index=False)
print("Particle sizes with Feret diameters and percentages saved to particle_sizes_pile_1.csv \n")

#--------------------------------------------------------------------------------------------#
# Visualization

#combining front and back part and normalizing them
points = np.vstack((points_back, points_front))
global_min = np.min(points, axis=0)
points_back = points_back - global_min
points_front = points_front - global_min
points = np.vstack((points_back, points_front))


#Black and white model

# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
intensity_b = las_file_back.intensity
intensity_f = las_file_front.intensity
colors_b = np.vstack((intensity_b, intensity_b, intensity_b)).T / np.max(intensity_b)
colors_f = np.vstack((intensity_f, intensity_f, intensity_f)).T / np.max(intensity_f)
colors_bf = np.vstack((colors_b, colors_f))
point_cloud.colors = o3d.utility.Vector3dVector(colors_bf)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])

#Give random colours to the particles
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

# Visualize the labeled point cloud
o3d.visualization.draw_geometries([point_cloud])

#--------------------------------------------------------------------------------------#

#Colour the particles based on their size
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

#Colour back particles
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
        elif feret_mm < min_diameter_mm: #Invalid small particles
            colors_back[labels_back == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: #0-25% of the largest valid particle size
            colors_back[labels_back == label] = [0.5, 0.0, 0.5] #Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: #25-50%
            colors_back[labels_back == label] = [1.0, 0.5, 0.0] #Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: #50-75%
            colors_back[labels_back == label] = [1.0, 1.0, 0.0] #Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: #75-100%
            colors_back[labels_back == label] = [1.0, 0.0, 0.0] #Red


#Colour front particles
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
        elif feret_mm < min_diameter_mm: #Invalid small particles
            colors_front[labels_front == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: #0-20% of the largest valid particle size
            colors_front[labels_front == label] = [0.5, 0.0, 0.5] #Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: #20-40%
            colors_front[labels_front == label] = [1.0, 0.5, 0.0] #Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: #40-60%
            colors_front[labels_front == label] = [1.0, 1.0, 0.0] #Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: #60-80%
            colors_front[labels_front == label] = [1.0, 0.0, 0.0] #Red
            

colors = np.vstack((colors_back, colors_front))

# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

point_cloud.colors = o3d.utility.Vector3dVector(colors)

# Visualize the labeled point cloud
o3d.visualization.draw_geometries([point_cloud])

#-------------------------------------------------------------------------------------------------------#
def plot_psd(df, bin_size=0.001):
    df_sorted = df.sort_values('Feret Diameter (mm)').reset_index(drop=True)
    diameters_mm = df_sorted['Feret Diameter (mm)']
    custom_cumulative_pct = df_sorted['Cumulative %']

    max_diam = np.ceil(diameters_mm.max())
    bins = np.arange(0, max_diam + bin_size * 1000, bin_size * 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.hist(diameters_mm, bins=bins, edgecolor='black')
    ax1.set_xlabel('Particle Diameter (mm)')
    ax1.set_ylabel('Count')
    ax1.set_title('Particle Size Distribution')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.set_xlim(0, diameters_mm.max())

    ax2.plot(diameters_mm, custom_cumulative_pct, 'r-', marker='o')
    ax2.set_xlabel('Particle Diameter (mm)')
    ax2.set_ylabel('Cumulative Percentage (%)')
    ax2.set_title('Cumulative PSD')
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, diameters_mm.max())

    x20 = np.interp(20, custom_cumulative_pct, diameters_mm)
    x50 = np.interp(50, custom_cumulative_pct, diameters_mm)
    x80 = np.interp(80, custom_cumulative_pct, diameters_mm)


    stats_text = (
        f"Ballangen Pile 1\n"
        f"Eps_back: {eps_back}\n"
        f"Eps_front: {eps_front}\n"
        f"min_samples: {min_samples_back}\n"
        f"Total particles: {len(diameters_mm)}\n"
        f"X₂₀: {x20:.1f} mm\n"
        f"X₅₀: {x50:.1f} mm\n"
        f"X₈₀: {x80:.1f} mm\n"
        f"Largest diameter: {diameters_mm.max():.2f} mm"
    )
    ax2.text(0.99, 0.01, stats_text, transform=ax2.transAxes,
             ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('particle_size_distribution_pile_1.png', dpi=300)
    plt.show()

plot_psd(df)