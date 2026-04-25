# RADAR - NUMBER OF OBSTACLES IN EACH FRAME==============================================================================================
# ===============================================================================================================================
import json
import numpy as np
from scipy.spatial.transform import Rotation

def load_data_files():
    """Load all required data files"""
    print("Loading RADAR data...")
    with open('./PATH/radar_file.json', 'r') as f:
        lidar_data = json.load(f)

    print("Loading ego poses...")
    with open('./PATH/ego_pose.json', 'r') as f:
        ego_poses = {item['token']: item for item in json.load(f)}

    print("Loading annotations...")
    with open('./PATH/sample_annotation.json', 'r') as f:
        annotations_data = json.load(f)

    return lidar_data, ego_poses, annotations_data

def create_annotations_mapping(annotations_data):
    """Create mapping from sample token to annotations"""
    annotations_dict = {}
    for ann in annotations_data:
        sample_token = ann['sample_token']
        if sample_token not in annotations_dict:
            annotations_dict[sample_token] = []
        annotations_dict[sample_token].append(ann)
    return annotations_dict

def process_frames(lidar_data, ego_poses, annotations_dict):
    """Process all frames and count obstacles for each frame"""
    print(f"\n{'='*80}")
    print(f"PROCESSING ALL FRAMES FOR OBSTACLE COUNT")
    print(f"{'='*80}")

    output_data = []
    
    for entry_idx, entry in enumerate(lidar_data):
        print(f"Processing frame {entry_idx + 1}/{len(lidar_data)}: {entry['token']}")
        
        # Create output entry with basic information
        output_entry = {
            "token": entry['token'],
            "ego_pose_token": entry['ego_pose_token'],
            "sample_token": entry['sample_token'],
            "timestamp": entry['timestamp'],
            "sensor_type": entry['sensor_type'],
            "original_file": entry['original_file'],
            "calibration_token": entry['calibration_token'],
            "Number_Of_Obstacles": 0,  # Initialize obstacle count to 0
            "velocities_x": entry.get('velocities_x', []),
            "velocities_y": entry.get('velocities_y', []),
            "prev": entry.get('prev', ''),
            "next": entry.get('next', ''),
            "is_faulty":entry['is_faulty'],
            "faulty_instance_token":entry['faulty_instance_token'],
            "type_of_feature_Extraction":"radarFront_obstacle_count_faulty",
            "interpolation_info": entry.get('interpolation_info', {})
        }
        
        # Get ego pose for this frame
        ego_pose_token = entry['ego_pose_token']
        if ego_pose_token not in ego_poses:
            print(f"  Warning: Ego pose token not found")
            output_data.append(output_entry)
            continue
        
        # Get annotations for this sample
        sample_token = entry['sample_token']
        annotations = annotations_dict.get(sample_token, [])
        
        # Count obstacles
        num_obstacles = len(annotations)
        output_entry["Number_Of_Obstacles"] = num_obstacles
        
        print(f"  Found {num_obstacles} obstacles")
        
        output_data.append(output_entry)
    
    return output_data

def print_statistics(output_data):
    """Print summary statistics about the processed data"""
    print(f"\n{'='*80}")
    print(f"PROCESSING STATISTICS")
    print(f"{'='*80}")
    
    total_frames = len(output_data)
    frames_with_obstacles = sum(1 for entry in output_data if entry["Number_Of_Obstacles"] > 0)
    total_obstacles = sum(entry["Number_Of_Obstacles"] for entry in output_data)
    
    print(f"Total frames processed: {total_frames}")
    print(f"Frames with obstacles: {frames_with_obstacles}")
    print(f"Frames without obstacles: {total_frames - frames_with_obstacles}")
    print(f"Total obstacles across all frames: {total_obstacles}")
    
    if frames_with_obstacles > 0:
        avg_obstacles_per_frame = total_obstacles / frames_with_obstacles
        print(f"Average obstacles per frame (with obstacles): {avg_obstacles_per_frame:.2f}")
    
    # Show obstacle distribution
    obstacle_counts = {}
    for entry in output_data:
        count = entry["Number_Of_Obstacles"]
        if count not in obstacle_counts:
            obstacle_counts[count] = 0
        obstacle_counts[count] += 1
    
    print(f"\nObstacle count distribution:")
    for count in sorted(obstacle_counts.keys()):
        frames = obstacle_counts[count]
        percentage = (frames / total_frames) * 100
        print(f"  {count:2d} obstacles: {frames:3d} frames ({percentage:5.1f}%)")
    
    # Show examples of first few entries
    print(f"\nFirst 5 entries as examples:")
    for i in range(min(5, len(output_data))):
        entry = output_data[i]
        print(f"  Frame {i+1}: {entry['token'][:8]}... - {entry['Number_Of_Obstacles']} obstacles")

def save_output_file(output_data):
    """Save the processed data to output JSON file"""
    output_file = './PATH/Radar_Detected_Objects.json'
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Processing completed. Results saved for {len(output_data)} frames.")

def main():
    """Main function to orchestrate the obstacle counting process"""
    # Load data
    lidar_data, ego_poses, annotations_data = load_data_files()
    
    # Create annotations mapping
    annotations_dict = create_annotations_mapping(annotations_data)
    
    # Process all frames
    output_data = process_frames(lidar_data, ego_poses, annotations_dict)
    
    # Print statistics
    print_statistics(output_data)
    
    # Save output file
    save_output_file(output_data)
    
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
