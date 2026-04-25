# RADAR obstacle angular distribution ====================================================================================
# =======================================================================================================================
import json
import numpy as np
from scipy.spatial.transform import Rotation
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge, Rectangle
import matplotlib.colors as mcolors

def calculate_angle_and_distance(point, ego_translation):
    """Calculate angle (in degrees) and distance from point to ego vehicle"""
    # Calculate vector from ego to point
    vector = point - ego_translation
    
    # Calculate distance
    distance = np.linalg.norm(vector)
    
    # Calculate angle in radians (0 degrees = front, 90 degrees = left, -90 degrees = right)
    angle_rad = math.atan2(vector[1], vector[0])  # atan2(y, x)
    
    # Convert to degrees and normalize to 0-360 range
    angle_deg = math.degrees(angle_rad)
    angle_deg = angle_deg % 360
    
    return angle_deg, distance

def get_box_corners_global(ann, ego_translation, ego_rotation):
    """Calculate all 8 corners of a 3D bounding box in global coordinates"""
    center = np.array(ann['translation'])
    size = np.array(ann['size'])  # [width, length, height]
    rotation = np.array(ann['rotation'])
    
    w, l, h = size
    
    # Define local corners (before rotation and translation)
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    
    corners_local = np.vstack([x_corners, y_corners, z_corners]).T
    
    # Convert to global coordinates
    rot = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
    corners_global = rot.apply(corners_local) + center
    
    return corners_global

def calculate_sector_distribution_for_box(box_corners, ego_translation, sectors, max_distance=250):  # Changed to 250m
    """
    Calculate distribution of a box across all sectors.
    Returns a dictionary with ratio for each sector where the box is present.
    The sum of all ratios will be 1.0 (if box is completely within 250m range).
    """
    # Sample points within the box (simplified to 2D projection)
    x_coords = box_corners[:, 0]
    y_coords = box_corners[:, 1]
    
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    
    # Use adaptive sampling based on box size
    box_width = max_x - min_x
    box_height = max_y - min_y
    num_samples_per_dim = max(10, int(min(box_width, box_height) * 2))  # More samples for larger boxes
    num_samples_per_dim = min(num_samples_per_dim, 50)  # Limit maximum samples
    
    sector_samples = {f"{int(start)}-{int(end)}": 0 for start, end in sectors}
    total_valid_samples = 0
    
    # Sample grid points within the box
    for i in range(num_samples_per_dim):
        for j in range(num_samples_per_dim):
            # Generate sample point within box bounds
            x = min_x + (i / (num_samples_per_dim - 1)) * box_width
            y = min_y + (j / (num_samples_per_dim - 1)) * box_height
            sample_point = np.array([x, y, np.mean(box_corners[:, 2])])  # Use average z
            
            angle_deg, distance = calculate_angle_and_distance(sample_point, ego_translation)
            
            # Only count samples within 250m range
            if distance <= max_distance:
                total_valid_samples += 1
                
                # Find which sector this sample belongs to
                for start_angle, end_angle in sectors:
                    sector_key = f"{int(start_angle)}-{int(end_angle)}"
                    
                    if start_angle <= end_angle:
                        if start_angle <= angle_deg < end_angle:
                            sector_samples[sector_key] += 1
                            break
                    else:
                        # Handle wrap-around (e.g., 354-360 + 0-6)
                        if angle_deg >= start_angle or angle_deg < end_angle:
                            sector_samples[sector_key] += 1
                            break
    
    # Calculate ratios
    sector_ratios = {}
    if total_valid_samples > 0:
        for sector_key, count in sector_samples.items():
            sector_ratios[sector_key] = count / total_valid_samples
    
    return sector_ratios, total_valid_samples

def plot_sector_visualization(ego_translation, annotations, sectors, frame_info):
    """Create a detailed visualization of sectors and obstacles for the last frame"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Full view with sectors and obstacles
    ax1.set_aspect('equal')
    ax1.set_title(f'Sector Distribution - {frame_info["token"]}\nSample: {frame_info["sample_token"]}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Plot ego vehicle at origin (since we're using relative coordinates)
    ego_circle = plt.Circle((0, 0), 2, color='red', alpha=0.7, label='Ego Vehicle')
    ax1.add_patch(ego_circle)
    ax1.plot(0, 0, 'ro', markersize=8, label='Ego Center')
    
    # Plot 250m circle (Changed to 250m)
    circle = plt.Circle((0, 0), 250, color='blue', fill=False, linestyle='--', linewidth=1, alpha=0.5, label='250m Range')
    ax1.add_patch(circle)
    
    # Plot sectors with different colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    sector_obstacle_ratios = {f"{int(start)}-{int(end)}": 0.0 for start, end in sectors}
    
    for i, (start_angle, end_angle) in enumerate(sectors):
        color = colors[i % len(colors)]
        wedge = Wedge((0, 0), 250, start_angle, end_angle, alpha=0.1, color=color)  # Changed to 250m
        ax1.add_patch(wedge)
        
        # Add sector label at 200m (adjusted for 250m range)
        mid_angle = (start_angle + end_angle) / 2
        rad = math.radians(mid_angle)
        label_x = 200 * math.cos(rad)  # Changed to 200m
        label_y = 200 * math.sin(rad)  # Changed to 200m
        ax1.text(label_x, label_y, f'{int(start_angle)}-{int(end_angle)}', 
                fontsize=6, ha='center', va='center', rotation=mid_angle, 
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
    
    # Plot obstacles with actual sizes and calculate intersection ratios
    obstacle_angles = []
    obstacle_distances = []
    obstacle_sizes = []
    
    for ann_idx, ann in enumerate(annotations):
        box_center_global = np.array(ann['translation'])
        box_corners = get_box_corners_global(ann, ego_translation, np.array([1, 0, 0, 0]))
        
        # Convert to relative coordinates (ego at origin)
        box_center_relative = box_center_global - ego_translation
        box_corners_relative = box_corners - ego_translation
        
        angle_deg, distance = calculate_angle_and_distance(box_center_global, ego_translation)
        obstacle_angles.append(angle_deg)
        obstacle_distances.append(distance)
        obstacle_sizes.append(ann['size'])
        
        # Calculate distribution across all sectors for this obstacle
        box_sector_ratios, total_samples = calculate_sector_distribution_for_box(box_corners, ego_translation, sectors)
        
        # Add to overall sector ratios
        for sector_key, ratio in box_sector_ratios.items():
            sector_obstacle_ratios[sector_key] += ratio
        
        # Plot obstacle box
        x_coords = box_corners_relative[:, 0]
        y_coords = box_corners_relative[:, 1]
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        width = max_x - min_x
        height = max_y - min_y
        
        rect = Rectangle((min_x, min_y), width, height, 
                        linewidth=1, edgecolor='red', facecolor='red', alpha=0.3,
                        label=f'Obstacle {ann_idx+1}' if ann_idx < 5 else "")
        ax1.add_patch(rect)
        
        # Add obstacle info
        if ann_idx < 10:  # Label first 10 obstacles
            sectors_for_this_obstacle = [k for k, v in box_sector_ratios.items() if v > 0.01]
            sector_info = ",".join([k.split('-')[0] for k in sectors_for_this_obstacle[:3]])
            if len(sectors_for_this_obstacle) > 3:
                sector_info += "..."
                
            ax1.annotate(f'{angle_deg:.0f}°\n{distance:.0f}m\nSectors: {sector_info}', 
                        (box_center_relative[0], box_center_relative[1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=6, alpha=0.8,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.5))
    
    # Add coordinate axes
    ax1.arrow(0, 0, 75, 0, head_width=5, head_length=7, fc='green', ec='green', label='Front (0°)')  # Scaled for 250m
    ax1.arrow(0, 0, 0, 75, head_width=5, head_length=7, fc='green', ec='green', label='Left (90°)')  # Scaled for 250m
    ax1.text(80, 0, '0°', fontsize=10, color='green', weight='bold')
    ax1.text(0, 80, '90°', fontsize=10, color='green', weight='bold')
    ax1.text(-80, 0, '180°', fontsize=10, color='green', weight='bold')
    ax1.text(0, -80, '270°', fontsize=10, color='green', weight='bold')
    
    ax1.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax1.set_xlim(-300, 300)  # Increased for 250m range
    ax1.set_ylim(-300, 300)  # Increased for 250m range
    
    # Plot 2: Polar plot for better angular visualization
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_title('Obstacle Angular Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # Convert angles to radians for polar plot
    obstacle_angles_rad = [math.radians(angle) for angle in obstacle_angles]
    
    # Plot obstacles in polar coordinates (size represents actual box size)
    sizes = [np.sqrt(size[0]**2 + size[1]**2) * 50 for size in obstacle_sizes]  # Scale for visibility
    scatter = ax2.scatter(obstacle_angles_rad, obstacle_distances, 
                         c=obstacle_distances, cmap='viridis', s=sizes,
                         alpha=0.7, label='Obstacles')
    
    # Add colorbar for distances
    cbar = plt.colorbar(scatter, ax=ax2, pad=0.1)
    cbar.set_label('Distance (m)', rotation=270, labelpad=15)
    
    # Plot sector boundaries
    for start_angle, end_angle in sectors:
        start_rad = math.radians(start_angle)
        end_rad = math.radians(end_angle)
        ax2.plot([start_rad, start_rad], [0, 250], 'r-', alpha=0.3, linewidth=0.5)  # Changed to 250m
    
    ax2.set_theta_offset(math.pi/2)  # 0 degrees at top
    ax2.set_theta_direction(-1)  # clockwise
    ax2.set_rmax(250)  # Changed to 250m
    ax2.set_rlabel_position(22.5)
    ax2.grid(True)
    
    # Add sector information as text
    info_text = f"Frame: {frame_info['token']}\n"
    info_text += f"Sample: {frame_info['sample_token']}\n"
    info_text += f"Total Obstacles: {len(annotations)}\n"
    info_text += f"Ego Position: ({ego_translation[0]:.1f}, {ego_translation[1]:.1f})\n\n"
    info_text += "Top Sectors (normalized ratio):\n"
    
    sectors_with_obstacles = [(k, v) for k, v in sector_obstacle_ratios.items() if v > 0]
    for sector_key, ratio in sorted(sectors_with_obstacles, key=lambda x: x[1], reverse=True)[:10]:
        normalized_ratio = ratio / len(annotations) if len(annotations) > 0 else 0
        info_text += f"{sector_key}°: {normalized_ratio:.4f}\n"
    
    total_normalized_ratio = sum(sector_obstacle_ratios.values()) / len(annotations) if len(annotations) > 0 else 0
    info_text += f"\nTotal Normalized: {total_normalized_ratio:.3f}"
    
    ax2.text(1.2, 0.5, info_text, transform=ax2.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return sector_obstacle_ratios

# Load data
print("Loading RADAR data...")
with open('./PATH/Radar_file.json', 'r') as f:
    radar_data = json.load(f)

print("Loading ego poses...")
with open('./PATH/ego_pose.json', 'r') as f:
    ego_poses = {item['token']: item for item in json.load(f)}

print("Loading annotations...")
with open('./PATH/sample_annotation.json', 'r') as f:
    annotations_data = json.load(f)

# Create mapping from sample token to annotations
annotations_dict = {}
for ann in annotations_data:
    sample_token = ann['sample_token']
    if sample_token not in annotations_dict:
        annotations_dict[sample_token] = []
    annotations_dict[sample_token].append(ann)

# Define sectors (60 sectors of 6 degrees each)
num_sectors = 60
sector_angle = 360 / num_sectors
sectors = []

for i in range(num_sectors):
    start_angle = i * sector_angle
    end_angle = (i + 1) * sector_angle
    sectors.append((start_angle, end_angle))

# Process only the last frame for detailed visualization
if radar_data:
    last_entry = radar_data[-1]
    print(f"\n{'='*80}")
    print(f"PROCESSING LAST FRAME FOR VISUALIZATION")
    print(f"{'='*80}")
    
    # Get ego pose for the last frame
    ego_pose_token = last_entry['ego_pose_token']
    if ego_pose_token in ego_poses:
        ego_pose = ego_poses[ego_pose_token]
        ego_translation = np.array(ego_pose['translation'])
        
        # Get annotations for this sample
        sample_token = last_entry['sample_token']
        annotations = annotations_dict.get(sample_token, [])
        
        if annotations:
            print(f"Last Frame: {last_entry['token']}")
            print(f"Sample: {sample_token}")
            print(f"Ego Position: {ego_translation}")
            print(f"Total Annotations: {len(annotations)}")
            print(f"Number of Sectors: {len(sectors)}")
            
            # Detailed analysis of obstacles with sizes
            print(f"\nDetailed Obstacle Analysis with Sizes:")
            for ann_idx, ann in enumerate(annotations[:5]):  # First 5 obstacles
                box_center_global = np.array(ann['translation'])
                box_corners = get_box_corners_global(ann, ego_translation, np.array([1, 0, 0, 0]))
                angle_deg, distance = calculate_angle_and_distance(box_center_global, ego_translation)
                
                # Calculate sector distribution for this obstacle
                box_sector_ratios, total_samples = calculate_sector_distribution_for_box(box_corners, ego_translation, sectors)
                
                print(f"  Obstacle {ann_idx+1}:")
                print(f"    Position: {box_center_global}")
                print(f"    Size: {ann['size']} (w:{ann['size'][0]:.2f}, l:{ann['size'][1]:.2f}, h:{ann['size'][2]:.2f})")
                print(f"    Angle: {angle_deg:.1f}°")
                print(f"    Distance: {distance:.1f}m")
                print(f"    Samples: {total_samples}")
                print(f"    Sector Distribution: {sum(box_sector_ratios.values()):.3f} total")
                
                # Show top sectors for this obstacle
                top_sectors = sorted(box_sector_ratios.items(), key=lambda x: x[1], reverse=True)[:3]
                for sector_key, ratio in top_sectors:
                    print(f"      {sector_key}°: {ratio:.4f}")
                print()
            
            # Create visualization
            frame_info = {
                'token': last_entry['token'],
                'sample_token': sample_token,
                'ego_position': ego_translation.tolist()
            }
            
            sector_ratios = plot_sector_visualization(ego_translation, annotations, sectors, frame_info)
            
            # Print sector statistics
            print(f"\nSector Statistics for Last Frame:")
            total_ratio_sum = sum(sector_ratios.values())
            print(f"Total ratio sum: {total_ratio_sum:.3f} (should be close to {len(annotations)})")
            
            sectors_with_obstacles = [(k, v) for k, v in sector_ratios.items() if v > 0]
            print(f"Sectors with obstacles: {len(sectors_with_obstacles)}/{len(sectors)}")
            
            if sectors_with_obstacles:
                print("\nTop sectors by obstacle ratio:")
                for sector_key, ratio in sorted(sectors_with_obstacles, key=lambda x: x[1], reverse=True)[:10]:
                    normalized_ratio = ratio / len(annotations)
                    print(f"  {sector_key}°: {ratio:.4f} raw, {normalized_ratio:.4f} normalized")
            
        else:
            print(f"No annotations found for sample token: {sample_token}")
    else:
        print(f"Ego pose token {ego_pose_token} not found for last frame")
else:
    print("No RADAR data found!")

# Processing for all frames with improved sector assignment
print(f"\n{'='*80}")
print(f"PROCESSING ALL FRAMES FOR JSON OUTPUT")
print(f"{'='*80}")

output_data = []
for entry_idx, entry in enumerate(radar_data):
    print(f"Processing frame {entry_idx + 1}/{len(radar_data)}: {entry['token']}")
    
    output_entry = {
        "token": entry['token'],
        "ego_pose_token": entry['ego_pose_token'],
        "sample_token": entry['sample_token'],
        "timestamp": entry['timestamp'],
        "sensor_type": entry['sensor_type'],
        "original_file": entry['original_file'],
        "calibration_token": entry['calibration_token'],
        "Slice_Value": [],
        "velocities_x": entry.get('velocities_x', []),
        "velocities_y": entry.get('velocities_y', []),
        "prev": entry.get('prev', ''),
        "next": entry.get('next', ''),
        "is_faulty":entry['is_faulty'],
        "faulty_instance_token":entry['faulty_instance_token'],
        "type_of_feature_Extraction":"radarFront_obstacle_sector_distribution_faulty",
        "interpolation_info": entry.get('interpolation_info', {})
    }
    
    ego_pose_token = entry['ego_pose_token']
    if ego_pose_token not in ego_poses:
        output_data.append(output_entry)
        continue
        
    ego_pose = ego_poses[ego_pose_token]
    ego_translation = np.array(ego_pose['translation'])
    
    sample_token = entry['sample_token']
    annotations = annotations_dict.get(sample_token, [])
    
    if not annotations:
        output_data.append(output_entry)
        continue
    
    total_obstacles = len(annotations)
    sector_ratios = {f"{int(start)}-{int(end)}": 0.0 for start, end in sectors}
    
    # Calculate distribution for each obstacle across all sectors
    for ann in annotations:
        box_corners = get_box_corners_global(ann, ego_translation, np.array([1, 0, 0, 0]))
        box_sector_ratios, _ = calculate_sector_distribution_for_box(box_corners, ego_translation, sectors)
        
        # Add to overall sector ratios
        for sector_key, ratio in box_sector_ratios.items():
            sector_ratios[sector_key] += ratio
    
    # Normalize ratios by total obstacles to get percentages
    for sector_key, ratio_sum in sector_ratios.items():
        normalized_ratio = ratio_sum / total_obstacles if total_obstacles > 0 else 0
        output_entry["Slice_Value"].append({sector_key: round(normalized_ratio, 6)})
    
    output_data.append(output_entry)

# Save to output file
output_file = './PATH/radar_obstacle_angular_distribution.json'
print(f"\nSaving results to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Processing completed. Results saved for {len(output_data)} frames.")

# Final statistics
print(f"\nFinal Statistics:")
total_normalized_check = 0
total_frames_with_data = 0

for entry in output_data:
    frame_total = sum(slice_val for slice_val in entry['Slice_Value'] for slice_val in slice_val.values())
    if frame_total > 0:
        total_normalized_check += frame_total
        total_frames_with_data += 1

if total_frames_with_data > 0:
    print(f"Average normalized ratio per frame: {total_normalized_check/total_frames_with_data:.3f}")
    print(f"Frames with obstacle data: {total_frames_with_data}/{len(output_data)}")
