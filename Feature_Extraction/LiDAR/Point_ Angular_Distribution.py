# LIDAR Points Angular Distribution==============================================================================================
# ===============================================================================================================================
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def local_to_global(point, translation, rotation):
    """Convert coordinates from vehicle-centric system to global system"""
    rot = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
    return rot.apply(point) + translation

def calculate_angle_from_ego(point_local, ego_translation, ego_rotation):
    """Calculate angle of point relative to ego vehicle's forward direction"""
    # Convert point to global coordinates
    point_global = local_to_global(point_local, ego_translation, ego_rotation)
    
    # Calculate vector from ego to point
    vector_to_point = point_global - ego_translation
    
    # For 2D angle calculation, we only care about x and y coordinates
    vector_2d = vector_to_point[:2]
    
    # Calculate angle in radians (0° is forward/x-axis, increasing counterclockwise)
    angle_rad = np.arctan2(vector_2d[1], vector_2d[0])
    
    # Convert to degrees and normalize to 0-360 range
    angle_deg = np.degrees(angle_rad) % 360
    
    return angle_deg

def visualize_last_frame(entry, ego_pose, output_entry, lidar_points_global, angles):
    """Visualize the last frame with ego vehicle, lidar points, and angular slices"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    
    ego_x, ego_y = ego_pose['translation'][0], ego_pose['translation'][1]
    
    # Plot 1: Scatter plot of lidar points with angular slices
    # Extract all global points
    points_x = [p[0] for p in lidar_points_global]
    points_y = [p[1] for p in lidar_points_global]
    
    # Plot ego vehicle
    ax1.scatter(ego_x, ego_y, color='red', s=100, label='Ego Vehicle', marker='s')
    
    # Plot lidar points colored by angle
    scatter = ax1.scatter(points_x, points_y, c=angles, cmap='hsv', s=1, alpha=0.7)
    plt.colorbar(scatter, ax=ax1, label='Angle (degrees)')
    
    # Draw angular slices (lines every 6 degrees) and add labels
    radius = 100
    label_radius = 110  # Radius for angle labels
    
    for i in range(0, 360, 6):
        angle_rad = np.radians(i)
        end_x = ego_x + radius * np.cos(angle_rad)
        end_y = ego_y + radius * np.sin(angle_rad)
        ax1.plot([ego_x, end_x], [ego_y, end_y], 'gray', alpha=0.3, linewidth=0.5)
        
        # Add angle labels for every 30 degrees (5 slices)
        if i % 30 == 0:
            label_x = ego_x + label_radius * np.cos(angle_rad)
            label_y = ego_y + label_radius * np.sin(angle_rad)
            
            # Determine slice range
            start_angle = i
            end_angle = (i + 6) % 360
            if end_angle == 0:
                end_angle = 360
            
            slice_label = f"{start_angle}-{end_angle}°"
            ax1.text(label_x, label_y, slice_label, fontsize=8, 
                    ha='center', va='center', rotation=i-90,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # Draw circles at different distances (ONLY FOR VISUALIZATION - NOT USED IN CALCULATIONS)
    for dist in [25, 50, 75, 100]:
        circle = plt.Circle((ego_x, ego_y), dist, fill=False, color='blue', alpha=0.3, linestyle='--')
        ax1.add_patch(circle)
        ax1.text(ego_x + dist, ego_y, f'{dist}m', fontsize=8, color='blue')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title(f'LIDAR Points with Angular Slices\nFrame: {entry["token"][:8]}...\n(Circles are for visualization only - not used in calculations)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Polar plot of point distribution
    ax2 = plt.subplot(122, projection='polar')
    
    # Create histogram of angles
    hist, bin_edges = np.histogram(angles, bins=60, range=(0, 360))
    bin_centers = np.deg2rad(bin_edges[:-1] + 3)  # Center of each 6-degree bin
    
    # Plot bars
    bars = ax2.bar(bin_centers, hist, width=np.deg2rad(6), alpha=0.7, 
                   color=plt.cm.hsv(bin_centers/(2*np.pi)))
    
    ax2.set_theta_zero_location('N')  # 0° at top
    ax2.set_theta_direction(-1)  # Clockwise
    ax2.set_title('Angular Distribution of LIDAR Points\n(Polar Plot)')
    ax2.grid(True)
    
    # Add angle labels to polar plot
    for i, (angle_rad, count) in enumerate(zip(bin_centers, hist)):
        if count > 0 and i % 5 == 0:  # Label every 5th slice
            ax2.text(angle_rad, max(hist)*1.1, f'{i*6}°', 
                    ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('./PATH/file_name.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS OF LAST FRAME")
    print(f"{'='*80}")
    print(f"Frame Token: {entry['token']}")
    print(f"Ego Position: ({ego_x:.2f}, {ego_y:.2f})")
    print(f"Total LIDAR Points: {len(lidar_points_global)}")
    print(f"Points with valid angles: {len(angles)}")
    
    # Show top 10 slices with most points
    slice_data = []
    for slice_val in output_entry['Slice_Value']:
        for slice_key, ratio in slice_val.items():
            count = int(ratio * len(lidar_points_global))
            slice_data.append((slice_key, ratio, count))
    
    slice_data.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop 10 Angular Slices by Point Count:")
    for i, (slice_key, ratio, count) in enumerate(slice_data[:10]):
        print(f"  {i+1:2d}. {slice_key}°: {count:4d} points ({ratio*100:6.2f}%)")
    
    # Show front/back/left/right statistics
    front_slices = [0, 1, 2, 58, 59]  # -12° to +12°
    left_slices = list(range(12, 18))  # 72° to 108°
    right_slices = list(range(42, 48))  # 252° to 288°
    back_slices = list(range(27, 33))  # 162° to 198°
    
    front_points = sum(slice_data[i][2] for i in front_slices if i < len(slice_data))
    left_points = sum(slice_data[i][2] for i in left_slices if i < len(slice_data))
    right_points = sum(slice_data[i][2] for i in right_slices if i < len(slice_data))
    back_points = sum(slice_data[i][2] for i in back_slices if i < len(slice_data))
    
    print(f"\nDirectional Distribution:")
    print(f"  Front (-12° to +12°): {front_points:4d} points ({front_points/len(angles)*100:5.1f}%)")
    print(f"  Left   (72° to 108°): {left_points:4d} points ({left_points/len(angles)*100:5.1f}%)")
    print(f"  Back  (162° to 198°): {back_points:4d} points ({back_points/len(angles)*100:5.1f}%)")
    print(f"  Right (252° to 288°): {right_points:4d} points ({right_points/len(angles)*100:5.1f}%)")
    
    # IMPORTANT NOTE ABOUT CIRCLES
    print(f"\n{'!'*80}")
    print("IMPORTANT NOTE ABOUT THE CIRCLES IN VISUALIZATION:")
    print("The circles at 25m, 50m, 75m, and 100m are ONLY for visual reference.")
    print("They are NOT used in the actual point distribution calculations.")
    print("The algorithm only considers ANGULAR distribution (0-360° in 6° slices).")
    print("Distance filtering is NOT applied - all points within LiDAR range are included.")
    print(f"{'!'*80}")

# Load LIDAR data
print("Loading LIDAR data...")
with open('./PATH/LiDAR_file.json', 'r') as f:
    lidar_data = json.load(f)

# Load ego poses
print("Loading ego poses...")
with open('./PATH/ego_pose.json', 'r') as f:
    ego_poses = {item['token']: item for item in json.load(f)}

# Define angle slices (60 slices of 6 degrees each)
angle_slices = []
for i in range(60):
    start_angle = i * 6
    end_angle = (i + 1) * 6
    angle_slices.append((start_angle, end_angle))

# Create output data structure
output_data = []
last_frame_data = None

print("Processing frames and calculating angular distributions...")

for entry in lidar_data:
    # Create output entry
    output_entry = {
        "token": entry['token'],
        "ego_pose_token": entry['ego_pose_token'],
        "sample_token": entry['sample_token'],
        "timestamp": entry['timestamp'],
        "sensor_type": entry['sensor_type'],
        "original_file": entry['original_file'],
        "calibration_token": entry['calibration_token'],
        "Slice_Value": [],
        "prev": entry.get('prev', ''),
        "next": entry.get('next', ''),
        "is_faulty":entry['is_faulty'],
        "faulty_instance_token":entry['faulty_instance_token'],
        "type_of_feature_Extraction":"lidar_points_sector_distribution_faulty",
        "interpolation_info": entry.get('interpolation_info', {})
    }
    
    # Get ego pose for this frame
    ego_pose_token = entry['ego_pose_token']
    if ego_pose_token not in ego_poses:
        print(f"Warning: Ego pose token {ego_pose_token} not found. Skipping frame {entry['token']}")
        output_data.append(output_entry)
        continue
        
    ego_pose = ego_poses[ego_pose_token]
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = np.array(ego_pose['rotation'])
    
    # Process lidar points
    raw_points = entry['calibrated_points']
    if len(raw_points) == 0:
        print(f"No lidar points in frame {entry['token']}")
        output_data.append(output_entry)
        continue
    
    # Initialize counters for each angle slice
    slice_counts = {f"{start}-{end}": 0 for start, end in angle_slices}
    
    # Count points in each angle slice
    total_points = len(raw_points)
    lidar_points_global = []
    angles = []
    
    for point in raw_points:
        if len(point) != 4:  # Skip if point doesn't have [x, y, z, instance_token]
            continue
            
        # Extract coordinates (first 3 elements)
        point_coords = np.array(point[:3])
        
        # Calculate angle relative to ego vehicle
        angle = calculate_angle_from_ego(point_coords, ego_translation, ego_rotation)
        
        # Store for visualization
        point_global = local_to_global(point_coords, ego_translation, ego_rotation)
        lidar_points_global.append(point_global)
        angles.append(angle)
        
        # Find which slice this angle belongs to
        for start, end in angle_slices:
            if start <= angle < end:
                slice_key = f"{start}-{end}"
                slice_counts[slice_key] += 1
                break
        # Handle the edge case where angle is exactly 360 degrees
            elif angle == 360:
                slice_key = "354-360"
                slice_counts[slice_key] += 1
    
    # Calculate ratios and add to output
    for slice_key, count in slice_counts.items():
        ratio = count / total_points if total_points > 0 else 0
        output_entry["Slice_Value"].append({
            slice_key: round(ratio, 6)  # Round to 6 decimal places for cleaner output
        })
    
    output_data.append(output_entry)
    
    # Store data for last frame visualization
    last_frame_data = (entry.copy(), ego_pose.copy(), output_entry.copy(), 
                      lidar_points_global.copy(), angles.copy())
    
    print(f"Processed frame {entry['token']}: {total_points} points, {len([s for s in slice_counts.values() if s > 0])} slices with points")

# Save to output file
output_file = './PATH/lidar_points_angular_distribution.json'
print(f"\nSaving results to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Processing completed. Results saved for {len(output_data)} frames.")

# Print summary statistics
total_frames = len(output_data)
frames_with_points = sum(1 for entry in output_data if any(slice_val for slice_val in entry['Slice_Value'] for val in slice_val.values() if val > 0))

print(f"\nSummary Statistics:")
print(f"Total frames processed: {total_frames}")
print(f"Frames with points: {frames_with_points}")
print(f"Angle slices: {len(angle_slices)} slices of 6° each")

# Show example of first few entries
print(f"\nFirst 3 entries as example:")
for i in range(min(3, len(output_data))):
    entry = output_data[i]
    print(f"Frame {i+1}: {entry['token']}")
    print(f"  Sample: {entry['sample_token']}")
    
    # Show only slices that have points
    slices_with_points = [slice_val for slice_val in entry['Slice_Value'] 
                         for key, val in slice_val.items() if val > 0]
    
    print(f"  Slices with points: {len(slices_with_points)}")
    for slice_val in slices_with_points[:5]:  # Show first 5 slices with points
        for slice_key, ratio in slice_val.items():
            print(f"    {slice_key}°: {ratio:.4f} ({ratio*100:.2f}%)")
    
    if len(slices_with_points) > 5:
        print(f"    ... and {len(slices_with_points) - 5} more slices")
    print()

# Visualize the last frame
if last_frame_data:
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATION FOR LAST FRAME...")
    print(f"{'='*80}")
    visualize_last_frame(*last_frame_data)
else:
    print("No frame data available for visualization.")

# Overall statistics across all frames
print("Overall Angular Distribution Statistics:")
all_ratios = {f"{start}-{end}": [] for start, end in angle_slices}

for entry in output_data:
    for slice_val in entry['Slice_Value']:
        for slice_key, ratio in slice_val.items():
            all_ratios[slice_key].append(ratio)

# Calculate average ratios for each slice
print("\nAverage distribution across all frames (showing slices with significant points):")
for slice_key, ratios in all_ratios.items():
    if ratios:  # Only show slices that have data
        avg_ratio = np.mean(ratios)
        max_ratio = np.max(ratios)
        frames_with_points = sum(1 for r in ratios if r > 0)
        
        if avg_ratio > 0.001:  # Only show slices with significant average
            print(f"  {slice_key}°: Avg={avg_ratio:.4f}, Max={max_ratio:.4f}, Frames={frames_with_points}")

# Final important note
print(f"\n{'='*80}")
print("FINAL IMPORTANT NOTE:")
print("The circles shown in visualization (25m, 50m, 75m, 100m) are ONLY for visual reference.")
print("They do NOT affect the point distribution calculations.")
print("The algorithm considers ALL LiDAR points regardless of distance.")
print("Only angular position (0-360°) is used for the 60 slices of 6° each.")
print(f"{'='*80}")
