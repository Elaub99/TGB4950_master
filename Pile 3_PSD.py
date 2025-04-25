import laspy
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, distance_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
#------------------------------------------------------------------------------------#
#back right part of the pile
# Open the .las file
las_file_back_right = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Ballangen/Scan/Pile 3_right_back.las")
points_back_right = np.vstack((las_file_back_right.x, las_file_back_right.y, las_file_back_right.z)).T

min_diameter_mm = 50
max_diameter_mm = 1500

eps_back_right = 0.015
min_samples_back_right = 7

dbscan_back_right = DBSCAN(eps_back_right, min_samples=min_samples_back_right)
labels_back_right = dbscan_back_right.fit_predict(points_back_right)
num_particles_back_right = len(np.unique(labels_back_right)) - 1
print(f"Number of particles identified in the back_right part of the pile: {num_particles_back_right}")
#------------------------------------------------------------------------------------#
#back left part of the pile
# Open the .las file
las_file_back_left = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Ballangen/Scan/Pile 3_left_back.las")
points_back_left = np.vstack((las_file_back_left.x, las_file_back_left.y, las_file_back_left.z)).T

min_diameter_mm = 50
max_diameter_mm = 1500

eps_back_left = 0.023
min_samples_back_left = 7

dbscan_back_left = DBSCAN(eps_back_left, min_samples=min_samples_back_left)
labels_back_left = dbscan_back_left.fit_predict(points_back_left)
num_particles_back_left = len(np.unique(labels_back_left)) - 1
print(f"Number of particles identified in the back left part of the pile: {num_particles_back_left}")

#------------------------------------------------------------------------------------#
#front_right
las_file_front_right = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Ballangen/Scan/Pile 3_right_front.las")
points_front_right = np.vstack((las_file_front_right.x, las_file_front_right.y, las_file_front_right.z)).T

eps_front_right = 0.007
min_samples_front_right = 7

dbscan_front_right = DBSCAN(eps_front_right, min_samples=min_samples_front_right)
labels_front_right = dbscan_front_right.fit_predict(points_front_right)
num_particles_front_right = len(np.unique(labels_front_right)) - 1
print(f"Number of particles identified in the front_right part of the pile: {num_particles_front_right}")

#------------------------------------------------------------------------------------#
#front_left
las_file_front_left = laspy.read("C:/Users/elaub/OneDrive - NTNU/Documents/5.aret/Master/Ballangen/Scan/Pile 3_left_front.las")
points_front_left = np.vstack((las_file_front_left.x, las_file_front_left.y, las_file_front_left.z)).T

eps_front_left = 0.012
min_samples_front_left = 7

dbscan_front_left = DBSCAN(eps_front_left, min_samples=min_samples_front_left)
labels_front_left = dbscan_front_left.fit_predict(points_front_left)
num_particles_front_left = len(np.unique(labels_front_left)) - 1
print(f"Number of particles identified in the front_left part of the pile: {num_particles_front_left}")

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

# Process right back part
for label in np.unique(labels_back_right):
    if label == -1:
        continue
    particle_points_back_right = points_back_right[labels_back_right == label]
    feret_mm = maximum_feret_diameter(particle_points_back_right) * 1000
    if feret_mm < min_diameter_mm or feret_mm > max_diameter_mm:
        continue
    results.append({
        "Particle ID": label,
        "Point Count": len(particle_points_back_right),
        "Feret Diameter (mm)": round(feret_mm, 3)
    })

# Process left back part
for label in np.unique(labels_back_left):
    if label == -1:
        continue
    particle_points_back_left = points_back_left[labels_back_left == label]
    feret_mm = maximum_feret_diameter(particle_points_back_left) * 1000
    if feret_mm < min_diameter_mm or feret_mm > max_diameter_mm:
        continue
    label += num_particles_back_right
    results.append({
        "Particle ID": label,
        "Point Count": len(particle_points_back_left),
        "Feret Diameter (mm)": round(feret_mm, 3)
    })

# Process right front part
for label in np.unique(labels_front_right):
    if label == -1:
        continue
    particle_points_front_right = points_front_right[labels_front_right == label]
    feret_mm = maximum_feret_diameter(particle_points_front_right) * 1000
    if feret_mm < min_diameter_mm or feret_mm > max_diameter_mm:
        continue
    label += num_particles_back_right + num_particles_back_left
    results.append({
        "Particle ID": label,
        "Point Count": len(particle_points_front_right),
        "Feret Diameter (mm)": round(feret_mm, 3)
    })
    
# Process front_left part
for label in np.unique(labels_front_left):
    if label == -1:
        continue
    particle_points_front_left = points_front_left[labels_front_left == label]
    feret_mm = maximum_feret_diameter(particle_points_front_left) * 1000
    if feret_mm < min_diameter_mm or feret_mm > max_diameter_mm:
        continue
    label += (num_particles_back_right + num_particles_back_left + num_particles_front_right)
    results.append({
        "Particle ID": label,
        "Point Count": len(particle_points_front_left),
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

print('Min particle size: ', min_diameter_mm, 'mm \n'
      'Max particle size: ', max_diameter_mm, 'mm \n'
      'The largest particle found is: ', largest_diameter, 'mm \n'
      'Number of valid particles are ', len(results))

# Save results
df = pd.DataFrame(results)
df.to_csv("particle_sizes_pile_3.csv", index=False)
print("Particle sizes with Feret diameters and percentages saved to particle_sizes_pile_3.csv \n")

#--------------------------------------------------------------------------------------------#
# Visualization

#combining front_right and back_right part and normalizing them
points = np.vstack((points_back_right, points_back_left, points_front_right, points_front_left))
global_min = np.min(points, axis=0)
points_back_right = points_back_right - global_min
points_back_left = points_back_left - global_min
points_front_right = points_front_right - global_min
points_front_left = points_front_left - global_min
points = np.vstack((points_back_right, points_back_left, points_front_right, points_front_left))


#Black and white model

# Create an Open3D point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
intensity_br = las_file_back_right.intensity
intensity_bl = las_file_back_left.intensity
intensity_fr = las_file_front_right.intensity
intensity_fl = las_file_front_left.intensity
colors_br = np.vstack((intensity_br, intensity_br, intensity_br)).T / np.max(intensity_br)
colors_bl = np.vstack((intensity_bl, intensity_bl, intensity_bl)).T / np.max(intensity_bl)
colors_fr = np.vstack((intensity_fr, intensity_fr, intensity_fr)).T / np.max(intensity_fr)
colors_fl = np.vstack((intensity_fl, intensity_fl, intensity_fl)).T / np.max(intensity_fl)
colors_all = np.vstack((colors_br, colors_bl, colors_fr, colors_fl))
point_cloud.colors = o3d.utility.Vector3dVector(colors_all)

# Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])

#Give random colours to the particles
colors_back_right = np.zeros((len(points_back_right), 3))
for label in np.unique(labels_back_right):
    if label == -1:  # Noise (unclustered points)
        colors_back_right[labels_back_right == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_back_right = points_back_right[labels_back_right == label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_back_right)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Large particles in light grey
            colors_back_right[labels_back_right == label] = [0.8, 0.8, 0.8]  # Light grey (RGB 80%)
        elif feret_mm < min_diameter_mm:
            colors_back_right[labels_back_right == label] = [0.1, 0.1, 0.1]    
        else:  # Normal particles get random colors
            colors_back_right[labels_back_right == label] = np.random.rand(3)  # Random color
            
colors_back_left = np.zeros((len(points_back_left), 3))
for label in np.unique(labels_back_left):
    if label == -1:  # Noise (unclustered points)
        colors_back_left[labels_back_left == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_back_left = points_back_left[labels_back_left == label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_back_left)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Large particles in light grey
            colors_back_left[labels_back_left == label] = [0.8, 0.8, 0.8]  # Light grey (RGB 80%)
        elif feret_mm < min_diameter_mm:
            colors_back_left[labels_back_left == label] = [0.1, 0.1, 0.1]    
        else:  # Normal particles get random colors
            colors_back_left[labels_back_left == label] = np.random.rand(3)  # Random color

colors_front_right= np.zeros((len(points_front_right), 3))            
for label in np.unique(labels_front_right):
    if label == -1:  # Noise (unclustered points)
        colors_front_right[labels_front_right == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_front_right = points_front_right[labels_front_right== label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_front_right)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Large particles in light grey
            colors_front_right[labels_front_right == label] = [0.8, 0.8, 0.8]  # Light grey (RGB 80%)
        elif feret_mm < min_diameter_mm:
            colors_front_right[labels_front_right == label] = [0.1, 0.1, 0.1]    
        else:  # Normal particles get random colors
            colors_front_right[labels_front_right == label] = np.random.rand(3)  # Random color
            
colors_front_left= np.zeros((len(points_front_left), 3))            
for label in np.unique(labels_front_left):
    if label == -1:  # Noise (unclustered points)
        colors_front_left[labels_front_left == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_front_left = points_front_left[labels_front_left == label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_front_left)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Large particles in light grey
            colors_front_left[labels_front_left == label] = [0.8, 0.8, 0.8]  # Light grey (RGB 80%)
        elif feret_mm < min_diameter_mm:
            colors_front_left[labels_front_left == label] = [0.1, 0.1, 0.1]    
        else:  # Normal particles get random colors
            colors_front_left[labels_front_left == label] = np.random.rand(3)  # Random color
            

colors = np.vstack((colors_back_right, colors_back_left, colors_front_right, colors_front_left))

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

#Colour back_right particles
colors_back_right = np.zeros((len(points_back_right), 3))
for label in np.unique(labels_back_right):
    if label == -1:  # Noise (unclustered points)
        colors_back_right[labels_back_right == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_back_right = points_back_right[labels_back_right == label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_back_right)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Invalid large particles
            colors_back_right[labels_back_right == label] = [0.8, 0.8, 0.8]  # Light grey
        elif feret_mm < min_diameter_mm: #Invalid small particles
            colors_back_right[labels_back_right == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: #0-25% of the largest valid particle size
            colors_back_right[labels_back_right == label] = [0.5, 0.0, 0.5] #Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: #25-50%
            colors_back_right[labels_back_right == label] = [1.0, 0.5, 0.0] #Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: #50-75%
            colors_back_right[labels_back_right == label] = [1.0, 1.0, 0.0] #Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: #75-100%
            colors_back_right[labels_back_right == label] = [1.0, 0.0, 0.0] #Red

#Colour back_left particles
colors_back_left = np.zeros((len(points_back_left), 3))
for label in np.unique(labels_back_left):
    if label == -1:  # Noise (unclustered points)
        colors_back_left[labels_back_left == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_back_left = points_back_left[labels_back_left == label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_back_left)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Invalid large particles
            colors_back_left[labels_back_left == label] = [0.8, 0.8, 0.8]  # Light grey
        elif feret_mm < min_diameter_mm: #Invalid small particles
            colors_back_left[labels_back_left == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: #0-25% of the largest valid particle size
            colors_back_left[labels_back_left == label] = [0.5, 0.0, 0.5] #Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: #25-50%
            colors_back_left[labels_back_left == label] = [1.0, 0.5, 0.0] #Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: #50-75%
            colors_back_left[labels_back_left == label] = [1.0, 1.0, 0.0] #Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: #75-100%
            colors_back_left[labels_back_left == label] = [1.0, 0.0, 0.0] #Red

#Colour front_right particles
colors_front_right= np.zeros((len(points_front_right), 3))            
for label in np.unique(labels_front_right):
    if label == -1:  # Noise (unclustered points)
        colors_front_right[labels_front_right == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_front_right = points_front_right[labels_front_right== label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_front_right)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Invalid large particles
            colors_front_right[labels_front_right == label] = [0.8, 0.8, 0.8]  # Light grey
        elif feret_mm < min_diameter_mm: #Invalid small particles
            colors_front_right[labels_front_right == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: #0-20% of the largest valid particle size
            colors_front_right[labels_front_right == label] = [0.5, 0.0, 0.5] #Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: #20-40%
            colors_front_right[labels_front_right == label] = [1.0, 0.5, 0.0] #Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: #40-60%
            colors_front_right[labels_front_right == label] = [1.0, 1.0, 0.0] #Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: #60-80%
            colors_front_right[labels_front_right == label] = [1.0, 0.0, 0.0] #Red
            
#Colour front_left particles
colors_front_left = np.zeros((len(points_front_left), 3))            
for label in np.unique(labels_front_left):
    if label == -1:  # Noise (unclustered points)
        colors_front_left[labels_front_left == label] = [0, 0, 0]  # Black for noise
    else:
        # Get the particle points
        particle_points_front_left = points_front_left[labels_front_left== label]
        # Calculate diameter
        feret_diameter = maximum_feret_diameter(particle_points_front_left)
        feret_mm = feret_diameter * 1000

        if feret_mm > max_diameter_mm:  # Invalid large particles
            colors_front_left[labels_front_left == label] = [0.8, 0.8, 0.8]  # Light grey
        elif feret_mm < min_diameter_mm: #Invalid small particles
            colors_front_left[labels_front_left == label] = [0, 0, 0] #Black
        elif (largest_diameter_25) >= feret_mm > min_diameter_mm: #0-20% of the largest valid particle size
            colors_front_left[labels_front_left == label] = [0.5, 0.0, 0.5] #Purple
        elif (largest_diameter_50) >= feret_mm > largest_diameter_25: #20-40%
            colors_front_left[labels_front_left == label] = [1.0, 0.5, 0.0] #Orange
        elif (largest_diameter_75) >= feret_mm > largest_diameter_50: #40-60%
            colors_front_left[labels_front_left == label] = [1.0, 1.0, 0.0] #Yellow
        elif (largest_diameter) >= feret_mm > largest_diameter_75: #60-80%
            colors_front_left[labels_front_left == label] = [1.0, 0.0, 0.0] #Red
            

colors = np.vstack((colors_back_right, colors_back_left, colors_front_right, colors_front_left))

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
        f"Ballangen Pile 3\n"
        f"Eps_back_right: {eps_back_right}\n"
        f"Eps_back_left: {eps_back_left}\n"
        f"Eps_front_right: {eps_front_right}\n"
        f"Eps_front_left: {eps_front_left}\n"
        f"min_samples: {min_samples_back_right}\n"
        f"Total particles: {len(diameters_mm)}\n"
        f"X₂₀: {x20:.1f} mm\n"
        f"X₅₀: {x50:.1f} mm\n"
        f"X₈₀: {x80:.1f} mm\n"
        f"Largest diameter: {diameters_mm.max():.2f} mm"
    )
    ax2.text(0.99, 0.01, stats_text, transform=ax2.transAxes,
             ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('particle_size_distribution_pile_3.png', dpi=300)
    plt.show()

plot_psd(df)