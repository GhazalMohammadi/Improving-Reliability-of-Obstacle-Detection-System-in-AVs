# LIDAR POINTS DISTANCE DISTRIBUTION====================================================================================
#================================================================================================================
import json
import numpy as np
from scipy.spatial.transform import Rotation

def local_to_global(point, translation, rotation):
    """Convert coordinates from vehicle-centric system to global system"""
    rot = Rotation.from_quat([rotation[1], rotation[2], rotation[3], rotation[0]])
    return rot.apply(point) + translation

def calculate_distance_to_ego(point_local, ego_translation, ego_rotation):
    """Calculate distance from point to ego vehicle"""
    # Convert point to global coordinates
    point_global = local_to_global(point_local, ego_translation, ego_rotation)
    
    # Calculate distance to ego vehicle
    distance = np.linalg.norm(point_global - ego_translation)
    return distance

# Load LIDAR data
print("Loading LIDAR data...")
with open('./PATH/LiDAR_file.json', 'r') as f:
    lidar_data = json.load(f)

# Load ego poses
print("Loading ego poses...")
with open('./PATH/ego_pose.json', 'r') as f:
    ego_poses = {item['token']: item for item in json.load(f)}

# Define distance ranges
distance_ranges = [
    (0, 5), (5, 10), (10, 15), (15, 20), (20, 25),
    (25, 30), (30, 35), (35, 40), (40, 45), (45, 50),
    (50, 55), (55, 60), (60, 65), (65, 70), (70, 75),
    (75, 80), (80, 85), (85, 90), (90, 95), (95, 100)
]

# Create output data structure
output_data = []

print("Processing frames and calculating distance distributions...")

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
        "is_faulty": entry['is_faulty'],
        "faulty_instance_token": entry['faulty_instance_token'],
        "type_of_feature_Extraction": "lidar_point_distance_distribution_faulty",
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
    
    # Initialize counters for each distance range
    range_counts = {f"{start}-{end}": 0 for start, end in distance_ranges}
    
    # Count points in each distance range
    total_points = len(raw_points)
    
    for point in raw_points:
        if len(point) != 4:  # Skip if point doesn't have [x, y, z, instance_token]
            continue
            
        # Extract coordinates (first 3 elements)
        point_coords = np.array(point[:3])
        
        # Calculate distance to ego vehicle
        distance = calculate_distance_to_ego(point_coords, ego_translation, ego_rotation)
        
        # Find which range this distance belongs to
        for start, end in distance_ranges:
            if start <= distance < end:
                range_key = f"{start}-{end}"
                range_counts[range_key] += 1
                break
    
    # Calculate ratios and add to output
    for range_key, count in range_counts.items():
        ratio = count / total_points if total_points > 0 else 0
        output_entry["Slice_Value"].append({
            range_key: round(ratio, 6)  # Round to 6 decimal places for cleaner output
        })
    
    output_data.append(output_entry)
    
    print(f"Processed frame {entry['token']}: {total_points} points, {len([r for r in range_counts.values() if r > 0])} ranges with points")

# Save to output file
output_file = './PATH/lidar_point_distance_distribution.json'

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
print(f"Distance ranges: {len(distance_ranges)} ranges from 0-100m")

# Show example of first few entries
print(f"\nFirst 3 entries as example:")
for i in range(min(3, len(output_data))):
    entry = output_data[i]
    print(f"Frame {i+1}: {entry['token']}")
    print(f"  Sample: {entry['sample_token']}")
    
    # Show only ranges that have points
    ranges_with_points = [slice_val for slice_val in entry['Slice_Value'] 
                         for key, val in slice_val.items() if val > 0]
    
    print(f"  Ranges with points: {len(ranges_with_points)}")
    for slice_val in ranges_with_points[:5]:  # Show first 5 ranges with points
        for range_key, ratio in slice_val.items():
            print(f"    {range_key}m: {ratio:.4f} ({ratio*100:.2f}%)")
    
    if len(ranges_with_points) > 5:
        print(f"    ... and {len(ranges_with_points) - 5} more ranges")
    
    # Show interpolation_info if it exists
    if entry['interpolation_info']:
        print(f"  Interpolation Info: {entry['interpolation_info']}")
    print()

# Overall statistics across all frames
print("Overall Distance Distribution Statistics:")
all_ratios = {f"{start}-{end}": [] for start, end in distance_ranges}

for entry in output_data:
    for slice_val in entry['Slice_Value']:
        for range_key, ratio in slice_val.items():
            all_ratios[range_key].append(ratio)

# Calculate average ratios for each range
print("\nAverage distribution across all frames:")
for range_key, ratios in all_ratios.items():
    if ratios:  # Only show ranges that have data
        avg_ratio = np.mean(ratios)
        max_ratio = np.max(ratios)
        frames_with_points = sum(1 for r in ratios if r > 0)
        
        if avg_ratio > 0.001:  # Only show ranges with significant average
            print(f"  {range_key}m: Avg={avg_ratio:.4f}, Max={max_ratio:.4f}, Frames={frames_with_points}")
