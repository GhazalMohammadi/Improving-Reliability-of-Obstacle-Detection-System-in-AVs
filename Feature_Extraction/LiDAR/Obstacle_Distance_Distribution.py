# LIDAR - obstacle_distance_distribution ================================================================================================
# ========================================================================================================================================
import json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def local_to_global(point, translation, rotation):
    """Convert coordinates from vehicle-centric system to global system"""
    rot = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
    return rot.apply(point) + translation

def calculate_distance_to_ego(point_global, ego_translation):
    """Calculate distance from point to ego vehicle"""
    return np.linalg.norm(point_global - ego_translation)

def calculate_obstacle_distance(annotation, ego_translation, ego_rotation):
    """
    Calculate the minimum distance from obstacle to ego vehicle.
    This calculates the theoretical minimum distance from any point on the obstacle to ego.
    """
    center = np.array(annotation['translation'])
    size = np.array(annotation['size'])  # [width, length, height]
    rotation = np.array(annotation['rotation'])
    
    w, l, h = size
    
    # Define local corners of the bounding box
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners_local = np.vstack([x_corners, y_corners, z_corners]).T
    
    # Convert to global coordinates
    rot = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
    corners_global = rot.apply(corners_local) + center
    
    # Calculate distance for each corner and return the minimum
    distances = [calculate_distance_to_ego(corner, ego_translation) for corner in corners_global]
    min_distance = min(distances)
    
    return min_distance

def visualize_last_frame(ego_translation, ego_rotation, annotations, distance_ranges, frame_info):
    """Visualize the last frame with ego vehicle, obstacles, and distance circles"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Top-down view with obstacles and distance circles
    ax1.set_aspect('equal')
    ax1.set_title(f'Obstacle Distance Distribution - Frame: {frame_info["token"][:8]}...\nSample: {frame_info["sample_token"][:8]}...', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.grid(True, alpha=0.3)
    
    # Plot ego vehicle at its global position
    ego_x, ego_y = ego_translation[0], ego_translation[1]
    ax1.scatter(ego_x, ego_y, color='red', s=100, label='Ego Vehicle', marker='s', zorder=5)
    
    # Plot distance circles
    colors = plt.cm.viridis(np.linspace(0, 1, len(distance_ranges)))
    for i, (start, end) in enumerate(distance_ranges):
        # Draw outer circle for each range
        circle = plt.Circle((ego_x, ego_y), end, fill=False, 
                          color=colors[i], alpha=0.6, linewidth=1, 
                          linestyle='--', label=f'{start}-{end}m')
        ax1.add_patch(circle)
        
        # Add distance labels
        if i % 4 == 0:  # Label every 4th circle to avoid clutter
            ax1.text(ego_x + end + 2, ego_y, f'{end}m', 
                    fontsize=8, color=colors[i], ha='left', va='center')
    
    # Plot obstacles with different colors based on distance
    obstacle_distances = []
    for i, ann in enumerate(annotations):
        distance = calculate_obstacle_distance(ann, ego_translation, ego_rotation)
        obstacle_distances.append(distance)
        
        center = np.array(ann['translation'])
        size = np.array(ann['size'])
        rotation = np.array(ann['rotation'])
        
        # Convert obstacle to global coordinates and plot
        w, l, h = size
        
        # Create bounding box corners (3D)
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [0, 0, 0, 0, 0, 0, 0, 0]  # Use z=0 for 2D visualization
        
        corners_local = np.vstack([x_corners, y_corners, z_corners]).T
        
        # Convert to global coordinates (3D)
        rot = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
        corners_global = rot.apply(corners_local) + center
        
        # Use only x,y coordinates for 2D visualization
        corners_global_2d = corners_global[:, :2]
        
        # Find which distance range this obstacle belongs to
        range_index = 0
        for j, (start, end) in enumerate(distance_ranges):
            if start <= distance < end:
                range_index = j
                break
        
        # Plot obstacle bounding box
        # Create polygon from corners (using first 4 corners for the base)
        from matplotlib.patches import Polygon
        poly = Polygon(corners_global_2d[:4], closed=True, 
                      alpha=0.7, color=colors[range_index], 
                      label=f'Obstacle {i+1}' if i < 5 else "")
        ax1.add_patch(poly)
        
        # Add obstacle info for first 10 obstacles
        if i < 10:
            ax1.annotate(f'{distance:.1f}m', 
                        (center[0], center[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
    
    # Set plot limits to show all obstacles and circles
    all_obstacle_x = [ann['translation'][0] for ann in annotations]
    all_obstacle_y = [ann['translation'][1] for ann in annotations]
    
    # Calculate appropriate margins based on data range
    x_range = max(all_obstacle_x) - min(all_obstacle_x) if all_obstacle_x else 0
    y_range = max(all_obstacle_y) - min(all_obstacle_y) if all_obstacle_y else 0
    
    x_margin = max(20, x_range * 0.1)
    y_margin = max(20, y_range * 0.1)
    
    x_min = min(all_obstacle_x + [ego_x]) - x_margin if all_obstacle_x else ego_x - 50
    x_max = max(all_obstacle_x + [ego_x]) + x_margin if all_obstacle_x else ego_x + 50
    y_min = min(all_obstacle_y + [ego_y]) - y_margin if all_obstacle_y else ego_y - 50
    y_max = max(all_obstacle_y + [ego_y]) + y_margin if all_obstacle_y else ego_y + 50
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Plot 2: Distance distribution histogram
    ax2.set_title('Obstacle Distance Distribution Histogram', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance from Ego Vehicle (meters)')
    ax2.set_ylabel('Number of Obstacles')
    
    # Create histogram data
    range_centers = [(start + end) / 2 for start, end in distance_ranges]
    range_counts = [0] * len(distance_ranges)
    
    for distance in obstacle_distances:
        for i, (start, end) in enumerate(distance_ranges):
            if start <= distance < end:
                range_counts[i] += 1
                break
    
    bars = ax2.bar(range_centers, range_counts, width=4.5, alpha=0.7, color=colors)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, range_counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=9)
    
    # Set x-axis ticks to match distance ranges
    ax2.set_xticks([center for center in range_centers])
    ax2.set_xticklabels([f'{start}-{end}' for start, end in distance_ranges], rotation=45)
    
    # Add frame information as text
    info_text = f"Frame: {frame_info['token']}\n"
    info_text += f"Sample: {frame_info['sample_token']}\n"
    info_text += f"Total Obstacles: {len(annotations)}\n"
    info_text += f"Ego Position: ({ego_x:.1f}, {ego_y:.1f})\n\n"
    info_text += "Distance Distribution:\n"
    
    for i, (start, end) in enumerate(distance_ranges):
        if range_counts[i] > 0:
            percentage = (range_counts[i] / len(annotations)) * 100
            info_text += f"{start}-{end}m: {range_counts[i]} ({percentage:.1f}%)\n"
    
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('./PATH/file_name.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return obstacle_distances

# Load LIDAR data
print("Loading LIDAR data...")
with open('./PATH/LiDAR_file.json', 'r') as f:
    lidar_data = json.load(f)

# Load ego poses
print("Loading ego poses...")
with open('./PATH/ego_pose.json', 'r') as f:
    ego_poses = {item['token']: item for item in json.load(f)}

# Load annotations
print("Loading annotations...")
with open('./PATH/sample_annotation.json', 'r') as f:
    annotations_data = json.load(f)

# Create mapping from sample token to annotations
print("Creating annotations mapping...")
annotations_dict = {}
for ann in annotations_data:
    sample_token = ann['sample_token']
    if sample_token not in annotations_dict:
        annotations_dict[sample_token] = []
    annotations_dict[sample_token].append(ann)

# Define distance ranges
distance_ranges = [
    (0, 5), (5, 10), (10, 15), (15, 20), (20, 25),
    (25, 30), (30, 35), (35, 40), (40, 45), (45, 50),
    (50, 55), (55, 60), (60, 65), (65, 70), (70, 75),
    (75, 80), (80, 85), (85, 90), (90, 95), (95, 100)
]

# Create output data structure
output_data = []
frame_obstacle_counts = {}  # Store actual obstacle counts per frame
last_frame_data = None  # Store data for last frame visualization

print("Processing frames and calculating obstacle distance distributions...")

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
        "type_of_feature_Extraction":"lidar_obstacle_distance_distribution_faulty",
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
    
    # Get annotations for this sample
    sample_token = entry['sample_token']
    annotations = annotations_dict.get(sample_token, [])
    
    if len(annotations) == 0:
        print(f"No obstacles found in frame {entry['token']}")
        output_data.append(output_entry)
        continue
    
    # Initialize counters for each distance range
    range_counts = {f"{start}-{end}": 0 for start, end in distance_ranges}
    
    # Count obstacles in each distance range
    total_obstacles = len(annotations)
    
    # Store actual obstacle count for this frame
    frame_obstacle_counts[entry['token']] = total_obstacles
    
    for ann in annotations:
        # Calculate minimum distance from obstacle to ego vehicle
        distance = calculate_obstacle_distance(ann, ego_translation, ego_rotation)
        
        # Find which range this distance belongs to
        for start, end in distance_ranges:
            if start <= distance < end:
                range_key = f"{start}-{end}"
                range_counts[range_key] += 1
                break
    
    # Calculate ratios and add to output
    for range_key, count in range_counts.items():
        ratio = count / total_obstacles if total_obstacles > 0 else 0
        output_entry["Slice_Value"].append({
            range_key: round(ratio, 6)  # Round to 6 decimal places for cleaner output
        })
    
    output_data.append(output_entry)
    
    # Store last frame data for visualization
    last_frame_data = {
        'entry': entry,
        'ego_pose': ego_pose,
        'ego_translation': ego_translation,
        'ego_rotation': ego_rotation,
        'annotations': annotations,
        'output_entry': output_entry
    }
    
    print(f"Processed frame {entry['token']}: {total_obstacles} obstacles, {len([r for r in range_counts.values() if r > 0])} ranges with obstacles")

# Save to output file
output_file = './PATH/LiDAR_obstacle_distance_distribution.json'
print(f"\nSaving results to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Processing completed. Results saved for {len(output_data)} frames.")

# Visualize last frame
if last_frame_data:
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATION FOR LAST FRAME...")
    print(f"{'='*80}")
    
    frame_info = {
        'token': last_frame_data['entry']['token'],
        'sample_token': last_frame_data['entry']['sample_token']
    }
    
    try:
        obstacle_distances = visualize_last_frame(
            last_frame_data['ego_translation'],
            last_frame_data['ego_rotation'],
            last_frame_data['annotations'],
            distance_ranges,
            frame_info
        )
        
        print(f"Last frame visualization completed!")
        print(f"Obstacle distances in last frame: {[f'{d:.1f}m' for d in obstacle_distances[:10]]}{'...' if len(obstacle_distances) > 10 else ''}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        print("Skipping visualization due to error...")
else:
    print("No frame data available for visualization.")

# Print summary statistics
total_frames = len(output_data)
frames_with_obstacles = sum(1 for entry in output_data if any(slice_val for slice_val in entry['Slice_Value'] for val in slice_val.values() if val > 0))

print(f"\nSummary Statistics:")
print(f"Total frames processed: {total_frames}")
print(f"Frames with obstacles: {frames_with_obstacles}")
print(f"Distance ranges: {len(distance_ranges)} ranges from 0-100m")

# Show example of first few entries
print(f"\nFirst 3 entries as example:")
for i in range(min(3, len(output_data))):
    entry = output_data[i]
    frame_token = entry['token']
    actual_obstacles = frame_obstacle_counts.get(frame_token, 0)
    
    print(f"Frame {i+1}: {frame_token}")
    print(f"  Sample: {entry['sample_token']}")
    print(f"  Total obstacles: {actual_obstacles}")
    
    # Show only ranges that have obstacles
    ranges_with_obstacles = [slice_val for slice_val in entry['Slice_Value'] 
                     for key, val in slice_val.items() if val > 0]
    
    print(f"  Ranges with obstacles: {len(ranges_with_obstacles)}")
    for slice_val in ranges_with_obstacles[:5]:  # Show first 5 ranges with obstacles
        for range_key, ratio in slice_val.items():
            actual_count = int(round(ratio * actual_obstacles))
            print(f"    {range_key}m: {ratio:.4f} ({ratio*100:.2f}%) - {actual_count} obstacles")
    
    if len(ranges_with_obstacles) > 5:
        print(f"    ... and {len(ranges_with_obstacles) - 5} more ranges")
    print()

# Overall statistics across all frames
print("Overall Obstacle Distance Distribution Statistics:")
all_ratios = {f"{start}-{end}": [] for start, end in distance_ranges}

for entry in output_data:
    for slice_val in entry['Slice_Value']:
        for range_key, ratio in slice_val.items():
            all_ratios[range_key].append(ratio)

# Calculate average ratios for each range
print("\nAverage distribution across all frames (showing ranges with significant obstacles):")
for range_key, ratios in all_ratios.items():
    if ratios:  # Only show ranges that have data
        avg_ratio = np.mean(ratios)
        max_ratio = np.max(ratios)
        frames_with_obstacles = sum(1 for r in ratios if r > 0)
        
        if avg_ratio > 0.001:  # Only show ranges with significant average
            print(f"  {range_key}m: Avg={avg_ratio:.4f}, Max={max_ratio:.4f}, Frames={frames_with_obstacles}")

# Additional detailed statistics - CORRECTED VERSION
print(f"\nDetailed Obstacle Statistics:")
total_obstacles_all_frames = sum(frame_obstacle_counts.values())
obstacle_distance_distribution = {f"{start}-{end}": 0 for start, end in distance_ranges}

# Calculate actual obstacle distribution
for entry in output_data:
    frame_token = entry['token']
    actual_obstacles = frame_obstacle_counts.get(frame_token, 0)
    
    for slice_val in entry['Slice_Value']:
        for range_key, ratio in slice_val.items():
            if ratio > 0:
                obstacle_count = int(round(ratio * actual_obstacles))
                obstacle_distance_distribution[range_key] += obstacle_count

print(f"Total obstacles across all frames: {total_obstacles_all_frames}")
print(f"\nObstacle distribution by distance range:")
for range_key, count in obstacle_distance_distribution.items():
    if count > 0:
        percentage = (count / total_obstacles_all_frames) * 100
        print(f"  {range_key}m: {count:4d} obstacles ({percentage:5.1f}%)")

# Verification
print(f"\nVerification:")
calculated_total = sum(obstacle_distance_distribution.values())
print(f"Sum of obstacles in all ranges: {calculated_total}")
print(f"Actual total obstacles: {total_obstacles_all_frames}")
print(f"Difference: {total_obstacles_all_frames - calculated_total} (should be 0 or very small due to rounding)")
