# Camera_Obstacle_Distance_Distribution======================================================================================
# ===========================================================================================================================
import json
import os
from collections import OrderedDict
import traceback

def create_distance_distribution(input_path, output_path):
    """
    Create distance distribution for camera obstacles
    
    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSON file
    
    Returns:
        Dictionary containing processed data
    """
    try:
        # Read the input JSON file
        print(f"Reading input file: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        # Define distance bins (0-5, 5-10, ..., 195-200)
        distance_bins = [f"{i}-{i+5}" for i in range(0, 200, 5)]
        
        # Initialize output structure
        output_data = OrderedDict()
        
        # Process each record - ensure we keep the original order
        total_records = len(data)
        print(f"Processing {total_records} records...")
        
        for idx, (image_name, record_data) in enumerate(data.items()):
            # Skip if record_data is not a dictionary
            if not isinstance(record_data, dict):
                print(f"Warning: Record {idx} is not a dictionary, skipping...")
                # Still create an empty record for this index
                output_record = OrderedDict([
                    ("token", ""),
                    ("sample_token", ""),
                    ("timestamp", 0),
                    ("sensor_type", "CAMERA"),
                    ("original_file", ""),
                    ("calibration_token", ""),
                    ("Slice_Value", []),
                    ("is_faulty", 0),
                    ("fault_type", ""),
                    ("type_of_feature_Extraction", "camera_obstacle_distance_distribution_faulty")
                ])
                output_data[str(idx)] = output_record
                continue
            
            # Get metadata
            metadata = record_data.get('metadata', {})
            
            # Get objects list - handle case where it doesn't exist
            objects_list = record_data.get('objects', [])
            if not isinstance(objects_list, list):
                objects_list = []
            
            total_objects = len(objects_list)
            
            # Initialize counts for each distance bin
            bin_counts = {bin_label: 0 for bin_label in distance_bins}
            
            # Count objects in each distance bin only if we have objects
            for obj in objects_list:
                if not isinstance(obj, dict):
                    continue
                
                distance = obj.get('distance')
                if distance is None:
                    continue
                
                try:
                    distance = float(distance)
                except (ValueError, TypeError):
                    continue
                
                # Find which bin this distance belongs to
                bin_index = int(distance // 5) * 5
                
                # Handle distances >= 200
                if distance >= 200:
                    bin_label = "195-200"
                else:
                    bin_label = f"{bin_index}-{bin_index + 5}"
                
                # Make sure bin_label is valid
                if bin_label in bin_counts:
                    bin_counts[bin_label] += 1
            
            # Calculate ratios - always create Slice_Value even if no objects
            slice_values = []
            for bin_label in distance_bins:
                ratio = 0.0
                if total_objects > 0:
                    ratio = bin_counts[bin_label] / total_objects
                slice_values.append({bin_label: round(ratio, 6)})
            
            # Create output record - match exact format from requirements
            output_record = OrderedDict([
                ("token", metadata.get("token", "")),
                ("sample_token", metadata.get("sample_token", "")),
                ("timestamp", metadata.get("timestamp", 0)),
                ("sensor_type", "CAMERA"),
                ("original_file", metadata.get("original_file", "")),
                ("calibration_token", metadata.get("calibration_token", "")),
                ("Slice_Value", slice_values),
                ("is_faulty", metadata.get("is_faulty", 0)),
                ("fault_type", metadata.get("fault_type", "")),
                ("type_of_feature_Extraction", "camera_obstacle_distance_distribution_faulty")
            ])
            
            # Add to output data with index as key
            output_data[str(idx)] = output_record
            
            # Progress indicator
            if (idx + 1) % 100 == 0 or (idx + 1) == total_records:
                print(f"Processed {idx + 1}/{total_records} records")
        
        # Write to output file
        print(f"Writing output to: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully processed {len(output_data)} records out of {total_records} total records.")
        
        # Verify count
        if len(output_data) != total_records:
            print(f"WARNING: Output count ({len(output_data)}) doesn't match input count ({total_records})")
            print("Checking which indices are missing...")
            missing_indices = []
            for i in range(total_records):
                if str(i) not in output_data:
                    missing_indices.append(i)
            if missing_indices:
                print(f"Missing indices: {missing_indices[:10]}{'...' if len(missing_indices) > 10 else ''}")
        
        return output_data
        
    except Exception as e:
        print(f"Error in create_distance_distribution: {str(e)}")
        print(traceback.format_exc())
        raise

def validate_output_format(output_data):
    """
    Validate that output format matches requirements
    
    Args:
        output_data: The output dictionary to validate
    """
    if not output_data:
        print("Warning: Output data is empty")
        return
    
    # Check a sample record
    sample_key = next(iter(output_data))
    sample_record = output_data[sample_key]
    
    required_fields = [
        "token", "sample_token", "timestamp", "sensor_type",
        "original_file", "calibration_token", "Slice_Value",
        "is_faulty", "fault_type", "type_of_feature_Extraction"
    ]
    
    print("\n=== Output Format Validation ===")
    for field in required_fields:
        if field in sample_record:
            print(f"Field '{field}' is present")
        else:
            print(f"Field '{field}' is missing")
    
    # Check Slice_Value structure
    if "Slice_Value" in sample_record:
        slice_values = sample_record["Slice_Value"]
        if isinstance(slice_values, list) and len(slice_values) > 0:
            print(f"Slice_Value contains {len(slice_values)} bins")
            # Should be 40 bins for 0-200 in 5-unit increments
            if len(slice_values) == 40:
                print("Correct number of distance bins (40)")
            else:
                print(f"Expected 40 bins, got {len(slice_values)}")
            
            # Check first few bins
            expected_first_bin = "0-5"
            if slice_values and expected_first_bin in slice_values[0]:
                print(f"First bin is correctly '{expected_first_bin}'")
            else:
                print(f"First bin format incorrect: {slice_values[0] if slice_values else 'Empty'}")
        else:
            print("Slice_Value is not a list or is empty")
    
    print("=== End Validation ===\n")

# Main execution
if __name__ == "__main__":
    # Paths
    input_file_path = r"C:\PATH\camera_obstacle_distances.json"
    output_file_path = r"C:\PATH\camera_obstacle_distance_distribution.json"
    
    print("=" * 60)
    print("Camera Obstacle Distance Distribution Processor")
    print("=" * 60)
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file_path):
            print(f"Error: Input file not found at: {input_file_path}")
            print("Please check the path and try again.")
            exit(1)
        
        # Process the data
        result = create_distance_distribution(input_file_path, output_file_path)
        
        # Validate output format
        validate_output_format(result)
        
        # Display summary
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Input file: {input_file_path}")
        print(f"Output file: {output_file_path}")
        print(f"Total records in input: 901 (expected)")
        print(f"Total records in output: {len(result)}")
        if len(result) == 901:
            print("SUCCESS: All 901 records processed!")
        else:
            print(f"WARNING: Only {len(result)} records processed, expected 901")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        print("Please check if the input file is valid JSON format.")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
