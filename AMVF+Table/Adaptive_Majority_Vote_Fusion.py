import json
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import cdist
from scipy.stats import zscore
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import re
from datetime import datetime
from scipy import stats

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------

def get_distance_for_instance(distance_list, instance_token):
    """Get distance for a specific instance from distance list"""
    if not distance_list:
        return None
        
    for item in distance_list:
        if isinstance(item, dict) and item.get('Instance_token') == instance_token:
            if item.get('Distance_to_Point') is not None:
                return float(item['Distance_to_Point'])
            else:
                return float(item['Theoretical_Closest'])
    return None

def extract_sensor_type_from_filepath(filepath):
    """Extract sensor type from filepath"""
    if not filepath:
        return None
        
    filepath = str(filepath).upper()
    
    if 'RADAR_FRONT' in filepath:
        if 'RADAR_FRONT_LEFT' in filepath:
            return 'RADAR_FRONT_LEFT'
        elif 'RADAR_FRONT_RIGHT' in filepath:
            return 'RADAR_FRONT_RIGHT'
        else:
            return 'RADAR_FRONT'
    elif 'CAM_FRONT' in filepath:
        if 'CAM_FRONT_LEFT' in filepath:
            return 'CAM_FRONT_LEFT'
        elif 'CAM_FRONT_RIGHT' in filepath:
            return 'CAM_FRONT_RIGHT'
        else:
            return 'CAM_FRONT'
    elif 'LIDAR' in filepath:
        return 'LIDAR_TOP'
    
    return None

def get_camera_distance_for_instance(record, instance_token):
    """Get distance for instance from camera record"""
    if not record or 'objects' not in record:
        return None
    
    for obj in record['objects']:
        if isinstance(obj, dict):
            if obj.get('instance_token') == instance_token:
                return obj.get('distance')
            
            if 'nuScenes_mapping' in obj:
                mapping = obj['nuScenes_mapping']
                if isinstance(mapping, dict) and mapping.get('instance_token') == instance_token:
                    return obj.get('distance')
    
    return None

def get_fault_status(record):
    """Get faulty status from record"""
    if not record:
        return False
        
    if 'is_faulty' in record:
        return bool(record['is_faulty'])
    elif 'metadata' in record and 'is_faulty' in record['metadata']:
        return bool(record['metadata']['is_faulty'])
    
    return False

def detect_outliers_iqr(data_dict):
    """
    Detect outliers using IQR method for sensor distances
    """
    if not data_dict:
        return {}
    
    distances = []
    sensor_info = []
    
    for sensor, data in data_dict.items():
        if data.get('available') and data.get('distance') is not None:
            distances.append(data['distance'])
            sensor_info.append((sensor, data['distance']))
    
    if len(distances) < 3:
        return {}
    
    q1 = np.percentile(distances, 25)
    q3 = np.percentile(distances, 75)
    iqr = q3 - q1
    
    if iqr == 0:
        return {}
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = {}
    for sensor, dist in sensor_info:
        if dist < lower_bound or dist > upper_bound:
            reason = "below lower bound" if dist < lower_bound else "above upper bound"
            deviation = abs(dist - np.median(distances)) / np.median(distances) * 100 if np.median(distances) != 0 else 0
            
            outliers[sensor] = {
                'distance': float(dist),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'median': float(np.median(distances)),
                'reason': reason,
                'deviation_percent': float(deviation),
                'method': 'iqr'
            }
    
    return outliers

def detect_outliers_zscore(data_dict, threshold=2.5):
    """
    Detect outliers using Z-score method
    """
    if not data_dict:
        return {}
    
    distances = []
    sensor_info = []
    
    for sensor, data in data_dict.items():
        if data.get('available') and data.get('distance') is not None:
            distances.append(data['distance'])
            sensor_info.append((sensor, data['distance']))
    
    if len(distances) < 3:
        return {}
    
    z_scores = np.abs(stats.zscore(distances))
    
    outliers = {}
    for (sensor, dist), z_score in zip(sensor_info, z_scores):
        if z_score > threshold:
            outliers[sensor] = {
                'distance': float(dist),
                'z_score': float(z_score),
                'threshold': float(threshold),
                'mean': float(np.mean(distances)),
                'std': float(np.std(distances)),
                'method': 'zscore'
            }
    
    return outliers

def detect_outliers_mad(data_dict, threshold=3.5):
    """
    Detect outliers using Median Absolute Deviation (MAD) method
    """
    if not data_dict:
        return {}
    
    distances = []
    sensor_info = []
    
    for sensor, data in data_dict.items():
        if data.get('available') and data.get('distance') is not None:
            distances.append(data['distance'])
            sensor_info.append((sensor, data['distance']))
    
    if len(distances) < 3:
        return {}
    
    median = np.median(distances)
    mad = np.median(np.abs(distances - median))
    
    if mad == 0:
        return {}
    
    modified_z_scores = 0.6745 * np.abs(distances - median) / mad
    
    outliers = {}
    for (sensor, dist), mod_z in zip(sensor_info, modified_z_scores):
        if mod_z > threshold:
            outliers[sensor] = {
                'distance': float(dist),
                'modified_z_score': float(mod_z),
                'threshold': float(threshold),
                'median': float(median),
                'mad': float(mad),
                'method': 'mad'
            }
    
    return outliers

# ----------------------------
# JSON SERIALIZATION HELPER
# ----------------------------

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, 
                             np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return super().default(obj)

# ----------------------------
# MAJORITY VOTE CORE CLASS WITH HISTORY AWARENESS
# ----------------------------

class MajorityVoteFusion:
    """Core class for Majority Vote fusion with clustering and history awareness"""
    
    def __init__(self, cluster_threshold=2.0, healthy_consensus_threshold=0.6, min_consensus_count=2):
        self.cluster_threshold = cluster_threshold
        self.healthy_consensus_threshold = healthy_consensus_threshold
        self.min_consensus_count = min_consensus_count
        self.faulty_weight_penalty = 0.5  # 50% penalty for faulty sensors in confidence calculation
        
        # History management
        self.history = {}  # instance_token -> list of fused distances
        self.history_window = 10  # Number of previous frames to consider
        
    def fuse_readings(self, readings, instance_token=None):
        """
        Fuse multiple readings using majority vote with clustering
        readings: list of dicts with keys: 'distance', 'weight', 'faulty', 'base_weight', 'sensor'
        instance_token: token for history tracking
        """
        if not readings:
            return None, {}
        
        if len(readings) == 1:
            return readings[0]['distance'], {'method': 'single_sensor', 'sensor': readings[0]['sensor']}
        
        # Step 1: Group readings into clusters based on distance similarity
        clusters = self._cluster_readings(readings)
        
        # Step 2: Analyze each cluster and calculate confidence
        cluster_analysis = []
        for i, cluster in enumerate(clusters):
            analysis = self._analyze_cluster(cluster)
            cluster_analysis.append(analysis)
        
        # Step 3: Calculate confidence for each cluster
        for i, cluster in enumerate(clusters):
            if i < len(cluster_analysis):
                cluster_analysis[i]['confidence'] = self._calculate_cluster_confidence(cluster)
        
        # Step 4: Get history for this instance if available
        history_distances = []
        if instance_token and instance_token in self.history:
            history_distances = self.history[instance_token]
        
        # Step 5: Select best cluster using majority vote rules with history awareness
        selected_cluster_idx = self._select_best_cluster_with_history(
            cluster_analysis, readings, history_distances
        )
        
        # Step 6: Calculate fused distance from selected cluster
        if selected_cluster_idx is not None:
            selected_cluster = clusters[selected_cluster_idx]
            selected_analysis = cluster_analysis[selected_cluster_idx]
            
            # Calculate weighted average from selected cluster
            total_weight = 0
            weighted_sum = 0
            
            for reading in selected_cluster:
                # Reduce weight for faulty readings in the cluster
                if reading['faulty']:
                    weight = reading['weight'] * 0.3
                else:
                    weight = reading['weight']
                
                weighted_sum += reading['distance'] * weight
                total_weight += weight
            
            if total_weight > 0:
                fused_distance = weighted_sum / total_weight
                
                fusion_details = {
                    'method': 'majority_vote',
                    'selected_cluster': selected_cluster_idx,
                    'clusters_count': len(clusters),
                    'cluster_analysis': cluster_analysis,
                    'selected_cluster_health_ratio': float(selected_analysis['health_ratio']),
                    'selected_cluster_sensor_count': len(selected_cluster),
                    'selected_cluster_faulty_count': selected_analysis['faulty_count'],
                    'selected_cluster_healthy_count': selected_analysis['healthy_count'],
                    'selected_cluster_confidence': float(selected_analysis.get('confidence', 0)),
                    'history_used': len(history_distances) > 0,
                    'history_size': len(history_distances)
                }
                
                # Update history if instance_token is provided
                if instance_token:
                    self._update_history(instance_token, fused_distance)
                
                return fused_distance, fusion_details
        
        # Fallback: Weighted average of all readings (with reduced weight for faulty)
        fallback_distance = self._weighted_average_fallback(readings)
        
        # Update history if instance_token is provided
        if instance_token and fallback_distance is not None:
            self._update_history(instance_token, fallback_distance)
        
        return fallback_distance, {'method': 'weighted_average_fallback'}
    
    def _cluster_readings(self, readings):
        """
        Group sensor readings into clusters based on distance similarity
        """
        if not readings:
            return []
        
        # Sort readings by distance
        sorted_readings = sorted(readings, key=lambda x: x['distance'])
        
        clusters = []
        current_cluster = [sorted_readings[0]]
        
        for i in range(1, len(sorted_readings)):
            current_reading = sorted_readings[i]
            last_reading = current_cluster[-1]
            
            # Check if this reading is similar to the last one in current cluster
            if abs(current_reading['distance'] - last_reading['distance']) <= self.cluster_threshold:
                current_cluster.append(current_reading)
            else:
                clusters.append(current_cluster)
                current_cluster = [current_reading]
        
        clusters.append(current_cluster)
        
        return clusters
    
    def _analyze_cluster(self, cluster):
        """
        Analyze a cluster of sensor readings
        """
        if not cluster:
            return None
        
        distances = [r['distance'] for r in cluster]
        faulty_count = sum(1 for r in cluster if r['faulty'])
        healthy_count = len(cluster) - faulty_count
        
        # Calculate cluster statistics
        mean_distance = float(np.mean(distances))
        std_distance = float(np.std(distances)) if len(distances) > 1 else 0
        
        # Calculate health ratio
        health_ratio = healthy_count / len(cluster) if len(cluster) > 0 else 0
        
        # Get sensor names
        sensors = [r['sensor'] for r in cluster]
        
        # Get additional metadata if available
        metadata = []
        for r in cluster:
            meta = {
                'sensor': r['sensor'],
                'distance': r['distance'],
                'faulty': bool(r['faulty']),  # Convert to Python bool
                'weight': float(r['weight']),
                'base_weight': float(r['base_weight'])
            }
            # Add sample_token if available
            if 'sample_token' in r:
                meta['sample_token'] = r['sample_token']
            if 'timestamp' in r:
                meta['timestamp'] = r['timestamp']
            metadata.append(meta)
        
        return {
            'sensor_count': len(cluster),
            'healthy_count': healthy_count,
            'faulty_count': faulty_count,
            'health_ratio': float(health_ratio),
            'mean_distance': mean_distance,
            'std_distance': std_distance,
            'sensors': sensors,
            'metadata': metadata
        }
    
    def _calculate_cluster_confidence(self, cluster):
        """
        Calculate confidence for a cluster with penalty for faulty sensors
        """
        if not cluster:
            return 0
        
        total_adjusted_weight = 0
        for reading in cluster:
            if reading['faulty']:
                # Apply 50% penalty for faulty sensors
                adjusted_weight = reading['base_weight'] * self.faulty_weight_penalty
            else:
                adjusted_weight = reading['base_weight']
            total_adjusted_weight += adjusted_weight
        
        # Normalize confidence
        confidence = total_adjusted_weight / len(cluster) if len(cluster) > 0 else 0
        
        return float(confidence)
    
    def _select_best_cluster_with_history(self, cluster_analysis, all_readings, history_distances):
        """
        Select the best cluster based on majority vote rules with history awareness
        """
        if not cluster_analysis:
            return None
        
        # Ensure confidence is calculated for each cluster
        clusters = self._cluster_readings(all_readings)
        for i, analysis in enumerate(cluster_analysis):
            if i < len(clusters):
                if 'confidence' not in analysis:
                    cluster = clusters[i]
                    analysis['confidence'] = self._calculate_cluster_confidence(cluster)
        
        # Rule 1: Prefer clusters with high health ratio
        healthy_clusters = []
        for i, analysis in enumerate(cluster_analysis):
            if analysis['health_ratio'] >= self.healthy_consensus_threshold:
                healthy_clusters.append((i, analysis))
        
        # Rule 2: If we have history, check for outliers and adjust cluster selection
        if history_distances and len(history_distances) >= 3:
            # Calculate adaptive threshold based on history
            adaptive_threshold = self._calculate_adaptive_threshold(history_distances)
            
            # Check each cluster against history
            history_compatible_clusters = []
            for i, analysis in enumerate(cluster_analysis):
                cluster_mean = analysis['mean_distance']
                if self._is_cluster_compatible_with_history(cluster_mean, history_distances, adaptive_threshold):
                    history_compatible_clusters.append((i, analysis))
            
            # If we have history-compatible clusters, use them
            if history_compatible_clusters:
                # First try to find history-compatible clusters among healthy clusters
                healthy_history_clusters = []
                for i, analysis in healthy_clusters:
                    for j, hist_analysis in history_compatible_clusters:
                        if i == j:
                            healthy_history_clusters.append((i, analysis))
                
                if healthy_history_clusters:
                    # Sort by sensor count (descending), then by confidence (descending)
                    healthy_history_clusters.sort(key=lambda x: (x[1]['sensor_count'], x[1]['confidence']), reverse=True)
                    return healthy_history_clusters[0][0]
                
                # If no healthy history clusters, use any history-compatible cluster
                history_compatible_clusters.sort(key=lambda x: (x[1]['sensor_count'], x[1]['confidence']), reverse=True)
                return history_compatible_clusters[0][0]
        
        # Rule 3: Original selection logic if no history or no compatible clusters
        if healthy_clusters:
            if len(healthy_clusters) == 1:
                return healthy_clusters[0][0]
            else:
                # Sort by sensor count (descending), then by confidence (descending)
                healthy_clusters.sort(key=lambda x: (x[1]['sensor_count'], x[1]['confidence']), reverse=True)
                return healthy_clusters[0][0]
        
        # Rule 4: If no cluster meets health threshold, check if any cluster has majority of sensors
        total_sensors = len(all_readings)
        clusters_with_majority = []
        for i, analysis in enumerate(cluster_analysis):
            if analysis['sensor_count'] >= max(self.min_consensus_count, total_sensors / 2):
                clusters_with_majority.append((i, analysis))
        
        if clusters_with_majority:
            # If multiple clusters have majority, select based on confidence
            if len(clusters_with_majority) == 1:
                return clusters_with_majority[0][0]
            else:
                # Sort by confidence (descending)
                clusters_with_majority.sort(key=lambda x: x[1]['confidence'], reverse=True)
                return clusters_with_majority[0][0]
        
        # Rule 5: If no clear majority, check if there's a cluster with reliable sensors
        for i, analysis in enumerate(cluster_analysis):
            # Check if cluster has high-reliability sensors (lidar or radar_front)
            high_rel_sensors = ['lidar', 'radar_front', 'radar_front_right', 'radar_front_left']
            cluster_sensors = analysis['sensors']
            high_rel_count = sum(1 for s in cluster_sensors if s in high_rel_sensors)
            
            if high_rel_count >= 2 and analysis['sensor_count'] >= 2:
                return i
        
        # Rule 6: If still no decision, use the cluster with highest confidence
        # First check if health ratios and sensor counts are equal
        all_have_same_health_ratio = len(set(a['health_ratio'] for a in cluster_analysis)) == 1
        all_have_same_sensor_count = len(set(a['sensor_count'] for a in cluster_analysis)) == 1
        
        if all_have_same_health_ratio and all_have_same_sensor_count:
            # Special case: health ratios and sensor counts are equal
            # Select based on confidence calculated with faulty penalty
            sorted_by_confidence = sorted(enumerate(cluster_analysis), 
                                         key=lambda x: x[1]['confidence'], reverse=True)
            return sorted_by_confidence[0][0]
        else:
            # Rule 7: If still no decision, use the largest cluster
            largest_cluster_idx = max(range(len(cluster_analysis)), 
                                     key=lambda i: cluster_analysis[i]['sensor_count'])
            
            return largest_cluster_idx
    
    def _calculate_adaptive_threshold(self, history_distances):
        """
        Calculate adaptive threshold based on history data variability
        """
        if len(history_distances) < 2:
            return 2.0  # Default threshold
        
        # Calculate rate of change in history
        changes = []
        for i in range(1, len(history_distances)):
            change = abs(history_distances[i] - history_distances[i-1])
            changes.append(change)
        
        # If no changes, return default
        if not changes:
            return 2.0
        
        # Calculate statistics of changes
        mean_change = float(np.mean(changes))
        std_change = float(np.std(changes))
        
        # Adaptive threshold: mean + 2*std, but at least 1.0 and at most 10.0
        adaptive_threshold = max(1.0, min(10.0, mean_change + 2 * std_change))
        
        return float(adaptive_threshold)
    
    def _is_cluster_compatible_with_history(self, cluster_mean, history_distances, threshold):
        """
        Check if a cluster is compatible with history
        """
        if not history_distances:
            return True
        
        # Calculate median of recent history (last 5 frames or all if less)
        recent_history = history_distances[-5:] if len(history_distances) >= 5 else history_distances
        history_median = float(np.median(recent_history))
        
        # Check if cluster mean is within threshold of history median
        return abs(cluster_mean - history_median) <= threshold
    
    def _update_history(self, instance_token, distance):
        """
        Update history for an instance token
        """
        if instance_token not in self.history:
            self.history[instance_token] = []
        
        self.history[instance_token].append(float(distance))
        
        # Keep only the last N distances
        if len(self.history[instance_token]) > self.history_window:
            self.history[instance_token] = self.history[instance_token][-self.history_window:]
    
    def _weighted_average_fallback(self, readings):
        """
        Fallback method: weighted average of all readings with penalty for faulty
        """
        if not readings:
            return None
        
        total_weight = 0
        weighted_sum = 0
        
        for reading in readings:
            # Apply penalty for faulty readings
            if reading['faulty']:
                weight = reading['weight'] * 0.3
            else:
                weight = reading['weight']
            
            weighted_sum += reading['distance'] * weight
            total_weight += weight
        
        if total_weight == 0:
            return None
        
        return weighted_sum / total_weight

# ----------------------------
# MULTI-SENSOR MAJORITY VOTE FUSION SYSTEM
# ----------------------------

class MultiSensorFuzzyFusion:
    def __init__(self):
        self.base_weights = {
            'lidar': 1.0,
            'radar_front': 0.7,
            'radar_front_right': 0.6,
            'radar_front_left': 0.5,
            'cam_front': 0.4,
            'cam_front_right': 0.3,
            'cam_front_left': 0.2
        }
        
        self.initialize_fuzzy_systems()
        
        self.outlier_detection_methods = ['iqr', 'mad']
        self.zscore_threshold = 3.0
        self.mad_threshold = 3.5
        
        # Outlier weight penalty factor (0.3 means outliers get 30% of their original weight)
        self.outlier_weight_penalty = 0.3
        
        # Majority Vote parameters with history awareness
        self.majority_vote = MajorityVoteFusion(
            cluster_threshold=2.0,
            healthy_consensus_threshold=0.6,
            min_consensus_count=2
        )
        
    def initialize_fuzzy_systems(self):
        """Initialize fuzzy logic systems for sensor fusion"""
        
        self.sensor_diff = ctrl.Antecedent(np.arange(0, 20, 0.1), 'sensor_difference')
        self.sensor_quality = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'sensor_quality')
        self.sensor_weight = ctrl.Consequent(np.arange(0, 1.1, 0.01), 'sensor_weight')
        
        self.sensor_diff['small'] = fuzz.trimf(self.sensor_diff.universe, [0, 0, 5])
        self.sensor_diff['medium'] = fuzz.trimf(self.sensor_diff.universe, [3, 8, 13])
        self.sensor_diff['large'] = fuzz.trimf(self.sensor_diff.universe, [10, 15, 20])
        
        self.sensor_quality['bad'] = fuzz.trimf(self.sensor_quality.universe, [0, 0, 0.5])
        self.sensor_quality['good'] = fuzz.trimf(self.sensor_quality.universe, [0.5, 1, 1])
        
        self.sensor_weight['very_low'] = fuzz.trimf(self.sensor_weight.universe, [0, 0.1, 0.3])
        self.sensor_weight['low'] = fuzz.trimf(self.sensor_weight.universe, [0.2, 0.4, 0.6])
        self.sensor_weight['medium'] = fuzz.trimf(self.sensor_weight.universe, [0.5, 0.7, 0.9])
        self.sensor_weight['high'] = fuzz.trimf(self.sensor_weight.universe, [0.8, 0.95, 1])
        
        rules = [
            ctrl.Rule(self.sensor_quality['good'] & self.sensor_diff['small'], 
                     self.sensor_weight['high']),
            ctrl.Rule(self.sensor_quality['good'] & self.sensor_diff['medium'], 
                     self.sensor_weight['medium']),
            ctrl.Rule(self.sensor_quality['good'] & self.sensor_diff['large'], 
                     self.sensor_weight['low']),
            
            ctrl.Rule(self.sensor_quality['bad'] & self.sensor_diff['small'], 
                     self.sensor_weight['medium']),
            ctrl.Rule(self.sensor_quality['bad'] & self.sensor_diff['medium'], 
                     self.sensor_weight['low']),
            ctrl.Rule(self.sensor_quality['bad'] & self.sensor_diff['large'], 
                     self.sensor_weight['very_low'])
        ]
        
        self.fusion_system = ctrl.ControlSystem(rules)
        self.fuser = ctrl.ControlSystemSimulation(self.fusion_system)
    
    def calculate_adjusted_weight(self, base_weight, distance, reference_distance, is_faulty):
        """
        Calculate adjusted weight for a sensor
        """
        if distance is None or reference_distance is None:
            return 0.0
        
        diff = abs(distance - reference_distance)
        
        try:
            self.fuser.input['sensor_difference'] = min(diff, 20)
            self.fuser.input['sensor_quality'] = 0.0 if is_faulty else 1.0
            self.fuser.compute()
            quality_factor = self.fuser.output['sensor_weight']
        except:
            if is_faulty:
                quality_factor = 0.3
            elif diff < 5:
                quality_factor = 0.9
            elif diff < 10:
                quality_factor = 0.6
            else:
                quality_factor = 0.4
        
        final_weight = base_weight * quality_factor
        
        return float(final_weight)
    
    def fuse_three_records(self, records_data, sensor_name):
        """
        Fuse three records (previous, current, next) using Majority Vote
        Returns: fused_distance, is_faulty, avg_timestamp, fusion_details
        """
        if not records_data:
            return None, False, None, {}
        
        # Prepare readings for Majority Vote fusion
        readings = []
        record_metadata = []  # Store metadata for each record
        
        for i, record_data in enumerate(records_data):
            if record_data['distance'] is not None:
                # Create a reading with metadata
                reading = {
                    'sensor': sensor_name,
                    'distance': float(record_data['distance']),
                    'weight': 0.7 if not record_data['faulty'] else 0.3,
                    'faulty': bool(record_data['faulty']),  # Convert to Python bool
                    'base_weight': float(self.base_weights.get(sensor_name.lower().replace('_', ''), 0.5)),
                    'timestamp': record_data['timestamp']
                }
                
                # Add sample_token if available
                if 'record' in record_data and 'sample_token' in record_data['record']:
                    reading['sample_token'] = record_data['record']['sample_token']
                
                readings.append(reading)
                
                # Store metadata for this record
                record_meta = {
                    'position': ['before', 'closest', 'after'][i] if i < 3 else f'position_{i}',
                    'distance': float(record_data['distance']),
                    'faulty': bool(record_data['faulty']),  # Convert to Python bool
                    'timestamp': record_data['timestamp']
                }
                
                if 'record' in record_data:
                    if 'sample_token' in record_data['record']:
                        record_meta['sample_token'] = record_data['record']['sample_token']
                    if 'original_file' in record_data['record']:
                        record_meta['original_file'] = record_data['record']['original_file']
                
                record_metadata.append(record_meta)
        
        if not readings:
            return None, False, None, {}
        
        # Use Majority Vote to fuse the three readings
        # Note: For triple fusion, we don't use instance token for history
        fused_distance, fusion_details = self.majority_vote.fuse_readings(readings)
        
        if fused_distance is not None:
            if fusion_details.get('method') == 'majority_vote':
                # Determine faulty or healthy based on selected cluster
                selected_cluster_idx = fusion_details.get('selected_cluster')
                if selected_cluster_idx is not None:
                    cluster_analysis = fusion_details.get('cluster_analysis', [])
                    if selected_cluster_idx < len(cluster_analysis):
                        selected_cluster = cluster_analysis[selected_cluster_idx]
                        faulty_count = selected_cluster.get('faulty_count', 0)
                        healthy_count = selected_cluster.get('healthy_count', 0)
                        
                        # If faulty > healthy = faulty, otherwise healthy
                        is_faulty = faulty_count > healthy_count
                        
                        # Store decision information
                        fusion_details['fault_determination'] = {
                            'method': 'selected_cluster_majority',
                            'selected_cluster_faulty_count': faulty_count,
                            'selected_cluster_healthy_count': healthy_count,
                            'is_faulty': bool(is_faulty),
                            'decision_reason': f"Faulty count ({faulty_count}) > Healthy count ({healthy_count}) = {faulty_count > healthy_count}"
                        }
                    else:
                        # Fallback: use previous method
                        faulty_count = sum(1 for record_data in records_data if record_data['faulty'])
                        total_count = len(records_data)
                        is_faulty = faulty_count >= (total_count / 2)
                        
                        fusion_details['fault_determination'] = {
                            'method': 'fallback_total_majority',
                            'total_faulty_count': faulty_count,
                            'total_count': total_count,
                            'is_faulty': bool(is_faulty),
                            'decision_reason': f"Fallback: Faulty count ({faulty_count}) >= Total/2 ({total_count}/2) = {faulty_count >= (total_count / 2)}"
                        }
                else:
                    # Fallback: use previous method
                    faulty_count = sum(1 for record_data in records_data if record_data['faulty'])
                    total_count = len(records_data)
                    is_faulty = faulty_count >= (total_count / 2)
                    
                    fusion_details['fault_determination'] = {
                        'method': 'fallback_total_majority',
                        'total_faulty_count': faulty_count,
                        'total_count': total_count,
                        'is_faulty': bool(is_faulty),
                        'decision_reason': f"Fallback: Faulty count ({faulty_count}) >= Total/2 ({total_count}/2) = {faulty_count >= (total_count / 2)}"
                    }
            else:
                # For other fusion methods, use previous method
                faulty_count = sum(1 for record_data in records_data if record_data['faulty'])
                total_count = len(records_data)
                is_faulty = faulty_count >= (total_count / 2)
                
                fusion_details['fault_determination'] = {
                    'method': 'total_majority',
                    'total_faulty_count': faulty_count,
                    'total_count': total_count,
                    'is_faulty': bool(is_faulty),
                    'decision_reason': f"Faulty count ({faulty_count}) >= Total/2 ({total_count}/2) = {faulty_count >= (total_count / 2)}"
                }
        else:
            # Fallback to weighted average
            total_weight = 0
            weighted_sum = 0
            for reading in readings:
                weight = reading['weight']
                weighted_sum += reading['distance'] * weight
                total_weight += weight
            
            if total_weight > 0:
                fused_distance = weighted_sum / total_weight
                fusion_details = {'method': 'weighted_average_fallback'}
                
                faulty_count = sum(1 for record_data in records_data if record_data['faulty'])
                total_count = len(records_data)
                is_faulty = faulty_count >= (total_count / 2)
                
                fusion_details['fault_determination'] = {
                    'method': 'weighted_average_fallback',
                    'total_faulty_count': faulty_count,
                    'total_count': total_count,
                    'is_faulty': bool(is_faulty),
                    'decision_reason': f"Faulty count ({faulty_count}) >= Total/2 ({total_count}/2) = {faulty_count >= (total_count / 2)}"
                }
            else:
                return None, False, None, {}
        
        # Calculate average timestamp
        timestamps = [record_data['timestamp'] for record_data in records_data if 'timestamp' in record_data]
        avg_timestamp = int(np.mean(timestamps)) if timestamps else None
        
        # Add record metadata to fusion details
        fusion_details['record_metadata'] = record_metadata
        fusion_details['num_records_fused'] = len(records_data)
        
        return float(fused_distance) if fused_distance is not None else None, is_faulty, avg_timestamp, fusion_details
    
    def fuse_all_sensors(self, sensor_data, instance_token=None):
        """
        Fuse all available sensor readings using majority vote with clustering
        instance_token: token for history tracking
        """
        available_sensors = [s for s in sensor_data if sensor_data[s]['available']]
        
        if not available_sensors:
            return None, {}
        
        if len(available_sensors) == 1:
            sensor = available_sensors[0]
            return sensor_data[sensor]['distance'], {'method': 'single_sensor', 'sensor': sensor}
        
        # Collect sensor readings with metadata
        readings = []
        for sensor in available_sensors:
            data = sensor_data[sensor]
            if data['distance'] is not None:
                # Apply outlier weight penalty if sensor is outlier
                if data.get('is_outlier', False):
                    effective_weight = data['weight'] * self.outlier_weight_penalty
                else:
                    effective_weight = data['weight']
                
                reading = {
                    'sensor': sensor,
                    'distance': float(data['distance']),
                    'weight': float(effective_weight),
                    'faulty': bool(data['faulty']),  # Convert to Python bool
                    'base_weight': float(self.base_weights.get(sensor, 0.5)),
                    'is_outlier': bool(data.get('is_outlier', False))  # Convert to Python bool
                }
                
                # Add metadata if available
                if 'sample_token' in data:
                    reading['sample_token'] = data['sample_token']
                if 'timestamp' in data:
                    reading['timestamp'] = data['timestamp']
                if 'record' in data:
                    reading['record'] = data['record']
                
                readings.append(reading)
        
        if not readings:
            return None, {}
        
        # Use Majority Vote to fuse all sensor readings with history awareness
        return self.majority_vote.fuse_readings(readings, instance_token)
    
    def detect_and_adjust_outliers(self, sensor_data):
        """
        Detect outliers and adjust their weights (instead of removing them)
        """
        filtered_data = sensor_data.copy()
        outlier_info = {}
        
        available_sensors = [s for s in sensor_data if sensor_data[s].get('available')]
        if len(available_sensors) < 3:
            # Mark all sensors as non-outliers
            for sensor in available_sensors:
                filtered_data[sensor]['is_outlier'] = False
                filtered_data[sensor]['outlier_weight_factor'] = 1.0
            return filtered_data, outlier_info
        
        all_outliers = {}
        
        iqr_outliers = detect_outliers_iqr(sensor_data)
        for sensor, info in iqr_outliers.items():
            if sensor not in all_outliers:
                all_outliers[sensor] = {'methods': ['iqr'], 'info': info}
            else:
                all_outliers[sensor]['methods'].append('iqr')
        
        mad_outliers = detect_outliers_mad(sensor_data, threshold=self.mad_threshold)
        for sensor, info in mad_outliers.items():
            if sensor not in all_outliers:
                all_outliers[sensor] = {'methods': ['mad'], 'info': info}
            else:
                all_outliers[sensor]['methods'].append('mad')
        
        # Mark outliers and adjust their weights
        for sensor in available_sensors:
            if sensor in all_outliers:
                # This sensor is an outlier
                filtered_data[sensor]['is_outlier'] = True
                filtered_data[sensor]['outlier_weight_factor'] = self.outlier_weight_penalty
                filtered_data[sensor]['outlier_reason'] = f"Detected by {', '.join(all_outliers[sensor]['methods'])}"
                
                outlier_info[sensor] = {
                    'methods': all_outliers[sensor]['methods'],
                    'distance': float(filtered_data[sensor]['distance']),
                    'original_weight': float(filtered_data[sensor]['weight']),
                    'adjusted_weight': float(filtered_data[sensor]['weight'] * self.outlier_weight_penalty),
                    'details': all_outliers[sensor]['info']
                }
            else:
                # This sensor is not an outlier
                filtered_data[sensor]['is_outlier'] = False
                filtered_data[sensor]['outlier_weight_factor'] = 1.0
        
        return filtered_data, outlier_info

# ----------------------------
# MAIN PROCESSING CLASS
# ----------------------------

class MultiSensorProcessor:
    def __init__(self):
        self.fusion_system = MultiSensorFuzzyFusion()
        self.time_threshold = 5000000
        # Store processing history for each instance
        self.processing_history = {}
        # Store history analysis for each instance
        self.history_analysis = {}
        
    def find_three_closest_records(self, records, target_timestamp, sensor_type, instance_token):
        """
        Find three records: closest, one before, and one after
        Returns: before_record, closest_record, after_record, status_msg, all_records_info
        """
        if not records:
            return None, None, None, "No records available", []
        
        filtered_records = []
        for record in records:
            if not isinstance(record, dict):
                continue
            
            filepath = record.get('original_file', '')
            detected_type = extract_sensor_type_from_filepath(filepath)
            if detected_type != sensor_type:
                continue
            
            if 'Distance' in record:
                distance = get_distance_for_instance(record['Distance'], instance_token)
                if distance is not None:
                    filtered_records.append({
                        'record': record,
                        'timestamp': record.get('timestamp'),
                        'distance': float(distance),
                        'faulty': bool(record.get('is_faulty', False))  # Convert to Python bool
                    })
        
        if not filtered_records:
            return None, None, None, f"No records with instance_token {instance_token}", []
        
        filtered_records.sort(key=lambda x: x['timestamp'])
        
        closest_record = None
        closest_diff = float('inf')
        closest_idx = -1
        
        for idx, rec_data in enumerate(filtered_records):
            diff = abs(rec_data['timestamp'] - target_timestamp)
            if diff < closest_diff:
                closest_diff = diff
                closest_record = rec_data
                closest_idx = idx
        
        if closest_diff > self.time_threshold:
            return None, None, None, f"No record within time threshold ({self.time_threshold/1e6:.2f}s)", []
        
        before_record = None
        after_record = None
        
        if closest_idx > 0:
            before_record = filtered_records[closest_idx - 1]
        
        if closest_idx < len(filtered_records) - 1:
            after_record = filtered_records[closest_idx + 1]
        
        # Collect info about all found records
        all_records_info = []
        if before_record:
            all_records_info.append({
                'position': 'before',
                'sample_token': before_record['record'].get('sample_token'),
                'timestamp': before_record['timestamp'],
                'distance': float(before_record['distance']),
                'faulty': bool(before_record['faulty'])  # Convert to Python bool
            })
        
        if closest_record:
            all_records_info.append({
                'position': 'closest',
                'sample_token': closest_record['record'].get('sample_token'),
                'timestamp': closest_record['timestamp'],
                'distance': float(closest_record['distance']),
                'faulty': bool(closest_record['faulty'])  # Convert to Python bool
            })
        
        if after_record:
            all_records_info.append({
                'position': 'after',
                'sample_token': after_record['record'].get('sample_token'),
                'timestamp': after_record['timestamp'],
                'distance': float(after_record['distance']),
                'faulty': bool(after_record['faulty'])  # Convert to Python bool
            })
        
        return before_record, closest_record, after_record, "Success", all_records_info
    
    def find_three_camera_records(self, camera_data, target_timestamp, camera_type, instance_token):
        """
        Find three camera records: closest, one before, and one after
        Returns: before_record, closest_record, after_record, status_msg, all_records_info
        """
        if not camera_data:
            return None, None, None, "No camera data available", []
        
        filtered_records = []
        
        for record_key, record in camera_data.items():
            if not isinstance(record, dict):
                continue
            
            metadata = record.get('metadata', {})
            filepath = metadata.get('original_file', '')
            detected_type = extract_sensor_type_from_filepath(filepath)
            
            if detected_type != camera_type:
                continue
            
            distance = get_camera_distance_for_instance(record, instance_token)
            if distance is None:
                continue
            
            timestamp = record.get('timestamp')
            if timestamp is None:
                timestamp = metadata.get('timestamp')
            
            if timestamp is None:
                continue
            
            filtered_records.append({
                'record_key': record_key,
                'record': record,
                'timestamp': timestamp,
                'distance': float(distance),
                'faulty': bool(get_fault_status(record))  # Convert to Python bool
            })
        
        if not filtered_records:
            return None, None, None, f"No camera records with instance_token {instance_token}", []
        
        filtered_records.sort(key=lambda x: x['timestamp'])
        
        closest_record = None
        closest_diff = float('inf')
        closest_idx = -1
        
        for idx, rec_data in enumerate(filtered_records):
            diff = abs(rec_data['timestamp'] - target_timestamp)
            if diff < closest_diff:
                closest_diff = diff
                closest_record = rec_data
                closest_idx = idx
        
        if closest_diff > self.time_threshold:
            return None, None, None, f"No camera record within time threshold ({self.time_threshold/1e6:.2f}s)", []
        
        before_record = None
        after_record = None
        
        if closest_idx > 0:
            before_record = filtered_records[closest_idx - 1]
        
        if closest_idx < len(filtered_records) - 1:
            after_record = filtered_records[closest_idx + 1]
        
        # Collect info about all found records
        all_records_info = []
        if before_record:
            all_records_info.append({
                'position': 'before',
                'sample_token': before_record['record_key'],
                'timestamp': before_record['timestamp'],
                'distance': float(before_record['distance']),
                'faulty': bool(before_record['faulty'])  # Convert to Python bool
            })
        
        if closest_record:
            all_records_info.append({
                'position': 'closest',
                'sample_token': closest_record['record_key'],
                'timestamp': closest_record['timestamp'],
                'distance': float(closest_record['distance']),
                'faulty': bool(closest_record['faulty'])  # Convert to Python bool
            })
        
        if after_record:
            all_records_info.append({
                'position': 'after',
                'sample_token': after_record['record_key'],
                'timestamp': after_record['timestamp'],
                'distance': float(after_record['distance']),
                'faulty': bool(after_record['faulty'])  # Convert to Python bool
            })
        
        return before_record, closest_record, after_record, "Success", all_records_info
    
    def _analyze_history_trend(self, instance_token):
        """
        Analyze history trend to understand if object is moving and how fast
        Returns: trend_info dict
        """
        if instance_token not in self.processing_history:
            return {
                'has_history': False,
                'history_size': 0,
                'is_moving': False,
                'movement_direction': 'unknown',
                'avg_change': 0,
                'max_change': 0,
                'trend_confidence': 0
            }
        
        history = self.processing_history[instance_token]
        
        if len(history) < 3:
            return {
                'has_history': True,
                'history_size': len(history),
                'is_moving': False,
                'movement_direction': 'unknown',
                'avg_change': 0,
                'max_change': 0,
                'trend_confidence': 0
            }
        
        # Calculate changes between consecutive frames
        changes = []
        for i in range(1, len(history)):
            change = history[i] - history[i-1]
            changes.append(change)
        
        avg_change = np.mean(changes)
        abs_changes = np.abs(changes)
        avg_abs_change = np.mean(abs_changes)
        max_abs_change = np.max(abs_changes)
        
        # Determine if object is moving significantly
        is_moving = avg_abs_change > 0.3  # Threshold for movement detection
        
        # Determine movement direction
        if avg_change < -0.1:
            movement_direction = 'approaching'
        elif avg_change > 0.1:
            movement_direction = 'receding'
        else:
            movement_direction = 'stable'
        
        # Calculate trend confidence based on consistency of changes
        if len(changes) >= 2:
            trend_confidence = 1.0 - (np.std(changes) / (avg_abs_change + 0.001))
            trend_confidence = max(0.0, min(1.0, trend_confidence))
        else:
            trend_confidence = 0.0
        
        return {
            'has_history': True,
            'history_size': len(history),
            'is_moving': bool(is_moving),
            'movement_direction': movement_direction,
            'avg_change': float(avg_change),
            'avg_abs_change': float(avg_abs_change),
            'max_change': float(max_abs_change),
            'trend_confidence': float(trend_confidence),
            'recent_changes': [float(c) for c in changes[-3:]] if changes else []
        }
    
    def _calculate_dynamic_threshold(self, instance_token, fused_distance):
        """
        Calculate dynamic threshold based on history trend and current context
        """
        trend_info = self._analyze_history_trend(instance_token)
        
        if not trend_info['has_history'] or trend_info['history_size'] < 3:
            # Not enough history, use default threshold
            return {
                'threshold': 2.0,
                'reason': 'insufficient_history',
                'trend_info': trend_info
            }
        
        if not trend_info['is_moving']:
            # Object is not moving much, use tighter threshold
            base_threshold = 1.5
            threshold_variation = 0.5 * (1.0 - trend_info['trend_confidence'])
            final_threshold = base_threshold + threshold_variation
            
            return {
                'threshold': float(final_threshold),
                'reason': 'stationary_object',
                'base_threshold': base_threshold,
                'threshold_variation': threshold_variation,
                'trend_info': trend_info
            }
        
        # Object is moving, use adaptive threshold
        avg_abs_change = trend_info['avg_abs_change']
        max_change = trend_info['max_change']
        trend_confidence = trend_info['trend_confidence']
        
        # Base threshold increases with movement speed
        base_threshold = max(2.0, min(5.0, 2.0 + avg_abs_change * 2))
        
        # Add variation based on trend confidence
        threshold_variation = 1.0 * (1.0 - trend_confidence)
        
        # Adjust based on maximum observed change
        if max_change > avg_abs_change * 2:
            # Unusual large change detected, be more conservative
            threshold_variation += 1.0
        
        final_threshold = base_threshold + threshold_variation
        
        # Cap the threshold at reasonable limits
        final_threshold = max(2.0, min(8.0, final_threshold))
        
        return {
            'threshold': float(final_threshold),
            'reason': 'moving_object',
            'base_threshold': base_threshold,
            'threshold_variation': threshold_variation,
            'avg_abs_change': avg_abs_change,
            'max_change': max_change,
            'trend_confidence': trend_confidence,
            'trend_info': trend_info
        }
    
    def _check_history_outlier(self, instance_token, fused_distance):
        """
        Check if the fused distance is an outlier compared to history
        Returns: (is_outlier, suggested_distance, details)
        """
        if instance_token not in self.processing_history:
            self.processing_history[instance_token] = []
        
        history = self.processing_history[instance_token]
        
        # If no history, cannot check
        if len(history) == 0:
            return False, fused_distance, {
                "reason": "No history",
                "history_size": 0,
                "threshold_info": {"threshold": 2.0, "reason": "no_history"}
            }
        
        # Get recent history (last 5 frames)
        recent_history = history[-5:] if len(history) >= 5 else history
        
        # Calculate statistics
        history_array = np.array(recent_history)
        median_distance = float(np.median(history_array))
        
        # For very short history (1 or 2 frames), use a simple threshold
        if len(recent_history) < 3:
            # Compare with the historical distances
            deviation = abs(fused_distance - median_distance)
            
            if len(recent_history) == 1:
                # Use a fixed threshold for single frame history
                adaptive_threshold = 3.0
                reason_suffix = " (single frame history)"
            else:
                # For 2 frames, calculate simple threshold
                std_distance = float(np.std(history_array))
                adaptive_threshold = max(2.0, 3 * std_distance) if std_distance > 0 else 2.0
                reason_suffix = " (two frame history)"
            
            is_outlier = deviation > adaptive_threshold
            
            details = {
                "history_size": len(recent_history),
                "history_values": [float(x) for x in recent_history],
                "median": median_distance,
                "adaptive_threshold": adaptive_threshold,
                "deviation": float(deviation),
                "is_outlier": bool(is_outlier),
                "threshold_info": {"threshold": adaptive_threshold, "reason": f"short_history{reason_suffix}"}
            }
            
            if is_outlier:
                suggested_distance = median_distance
                details["suggested_distance"] = suggested_distance
                details["reason"] = f"Deviation {deviation:.2f}m > threshold {adaptive_threshold:.2f}m{reason_suffix}"
            else:
                suggested_distance = fused_distance
                details["reason"] = f"Deviation {deviation:.2f}m <= threshold {adaptive_threshold:.2f}m{reason_suffix}"
            
            return is_outlier, suggested_distance, details
        
        # For 3 or more frames, use trend-aware analysis
        std_distance = float(np.std(history_array))
        
        # Calculate rate of change in history
        changes = np.abs(np.diff(recent_history))
        avg_change = float(np.mean(changes))
        max_change = float(np.max(changes))
        
        # Get dynamic threshold based on trend analysis
        threshold_info = self._calculate_dynamic_threshold(instance_token, fused_distance)
        adaptive_threshold = threshold_info['threshold']
        
        deviation = abs(fused_distance - median_distance)
        
        # Check if it's an outlier
        is_outlier = deviation > adaptive_threshold
        
        details = {
            "history_size": len(recent_history),
            "history_values": [float(x) for x in recent_history],
            "median": median_distance,
            "std": std_distance,
            "avg_change": avg_change,
            "max_change": max_change,
            "adaptive_threshold": adaptive_threshold,
            "deviation": float(deviation),
            "is_outlier": bool(is_outlier),
            "threshold_info": threshold_info
        }
        
        if is_outlier:
            # For outliers, we need to decide whether to use median or something else
            # Check if the deviation is in the expected direction based on trend
            trend_info = threshold_info['trend_info']
            
            if trend_info['is_moving']:
                # Object is moving, check if deviation is in the expected direction
                expected_change = trend_info['avg_change']
                actual_change = fused_distance - recent_history[-1]
                
                # If actual change is in opposite direction of expected trend, it's suspicious
                if expected_change > 0 and actual_change < -adaptive_threshold:
                    # Expected to increase but decreased significantly
                    suggested_distance = median_distance
                    details["reason"] = f"Unexpected decrease: deviation {deviation:.2f}m > threshold {adaptive_threshold:.2f}m, expected increase"
                elif expected_change < 0 and actual_change > adaptive_threshold:
                    # Expected to decrease but increased significantly
                    suggested_distance = median_distance
                    details["reason"] = f"Unexpected increase: deviation {deviation:.2f}m > threshold {adaptive_threshold:.2f}m, expected decrease"
                else:
                    # Deviation might be due to acceleration/deceleration
                    # Use weighted average between median and current
                    weight = 0.7  # Favor history more
                    suggested_distance = (weight * median_distance + (1 - weight) * fused_distance)
                    details["reason"] = f"Moving object: deviation {deviation:.2f}m > threshold {adaptive_threshold:.2f}m, using weighted correction"
            else:
                # Stationary object, use median
                suggested_distance = median_distance
                details["reason"] = f"Stationary object: deviation {deviation:.2f}m > threshold {adaptive_threshold:.2f}m"
            
            details["suggested_distance"] = suggested_distance
        else:
            suggested_distance = fused_distance
            details["reason"] = f"Deviation {deviation:.2f}m <= threshold {adaptive_threshold:.2f}m"
        
        return is_outlier, suggested_distance, details
    
    def _apply_history_correction(self, instance_token, fused_distance, majority_vote_details):
        """
        Apply history-based correction to fused distance
        Returns: (corrected_distance, corrected_details, correction_info)
        """
        # Check if this is an outlier compared to history
        is_outlier, suggested_distance, history_details = self._check_history_outlier(
            instance_token, fused_distance
        )
        
        if not is_outlier:
            # Not an outlier, keep the original result
            return fused_distance, majority_vote_details, {
                "history_correction_applied": False,
                "reason": "Not an outlier",
                "details": history_details
            }
        
        # It's an outlier, need to reconsider
        print(f"\n  [HISTORY ANALYSIS] Outlier detected!")
        print(f"    Original distance: {fused_distance:.2f}m")
        print(f"    History median: {history_details['median']:.2f}m")
        print(f"    Deviation: {history_details['deviation']:.2f}m")
        print(f"    Adaptive threshold: {history_details['adaptive_threshold']:.2f}m")
        print(f"    Reason: {history_details['reason']}")
        
        # Check threshold info for trend analysis
        threshold_info = history_details.get('threshold_info', {})
        if 'trend_info' in threshold_info:
            trend_info = threshold_info['trend_info']
            print(f"    Trend analysis:")
            print(f"      Is moving: {trend_info.get('is_moving', False)}")
            print(f"      Direction: {trend_info.get('movement_direction', 'unknown')}")
            print(f"      Avg change: {trend_info.get('avg_change', 0):.2f}m/frame")
            print(f"      Avg abs change: {trend_info.get('avg_abs_change', 0):.2f}m/frame")
        
        # Check if majority vote method can provide alternatives
        if majority_vote_details.get('method') == 'majority_vote':
            clusters = majority_vote_details.get('cluster_analysis', [])
            selected_cluster_idx = majority_vote_details.get('selected_cluster')
            
            if clusters and selected_cluster_idx is not None:
                # Find alternative clusters that are compatible with history
                history_median = history_details["median"]
                adaptive_threshold = history_details["adaptive_threshold"]
                
                compatible_clusters = []
                for idx, cluster in enumerate(clusters):
                    if idx == selected_cluster_idx:
                        continue
                    
                    cluster_mean = cluster['mean_distance']
                    deviation = abs(cluster_mean - history_median)
                    
                    if deviation <= adaptive_threshold:
                        # This cluster is compatible with history
                        compatible_clusters.append({
                            'idx': idx,
                            'mean_distance': cluster_mean,
                            'deviation': deviation,
                            'sensor_count': cluster['sensor_count'],
                            'health_ratio': cluster['health_ratio'],
                            'confidence': cluster.get('confidence', 0)
                        })
                
                if compatible_clusters:
                    # Sort compatible clusters by deviation from history (ascending), then by confidence (descending)
                    compatible_clusters.sort(key=lambda x: (x['deviation'], -x['confidence']))
                    
                    best_cluster = compatible_clusters[0]
                    corrected_distance = best_cluster['mean_distance']
                    
                    print(f"\n  [HISTORY CORRECTION] Found compatible cluster!")
                    print(f"    Original cluster {selected_cluster_idx}: {fused_distance:.2f}m")
                    print(f"    Compatible cluster {best_cluster['idx']}: {corrected_distance:.2f}m")
                    print(f"    Deviation from history: {best_cluster['deviation']:.2f}m")
                    print(f"    Cluster confidence: {best_cluster['confidence']:.2f}")
                    
                    # Update majority vote details
                    corrected_details = majority_vote_details.copy()
                    corrected_details['selected_cluster'] = best_cluster['idx']
                    corrected_details['selected_cluster_health_ratio'] = float(best_cluster['health_ratio'])
                    corrected_details['selected_cluster_sensor_count'] = best_cluster['sensor_count']
                    corrected_details['selected_cluster_confidence'] = float(best_cluster['confidence'])
                    corrected_details['original_selected_cluster'] = selected_cluster_idx
                    corrected_details['history_correction'] = {
                        'applied': True,
                        'reason': f'Outlier detected: original distance {fused_distance:.2f}m, '
                                 f'history median {history_median:.2f}m, deviation {history_details["deviation"]:.2f}m',
                        'history_details': history_details,
                        'corrected_distance': corrected_distance,
                        'original_distance': fused_distance,
                        'correction_method': 'alternative_cluster'
                    }
                    
                    return corrected_distance, corrected_details, {
                        "history_correction_applied": True,
                        "original_distance": fused_distance,
                        "corrected_distance": corrected_distance,
                        "original_cluster": selected_cluster_idx,
                        "corrected_cluster": best_cluster['idx'],
                        "correction_method": "alternative_cluster",
                        "details": history_details
                    }
        
        # If no compatible cluster found or not majority vote method, use the suggested distance
        corrected_distance = suggested_distance
        
        print(f"\n  [HISTORY CORRECTION] Using history-based suggestion")
        print(f"    Original distance: {fused_distance:.2f}m")
        print(f"    Corrected distance: {corrected_distance:.2f}m")
        
        # Update majority vote details to reflect the correction
        corrected_details = majority_vote_details.copy()
        corrected_details['history_correction'] = {
            'applied': True,
            'reason': f'Outlier detected: original distance {fused_distance:.2f}m, '
                     f'history median {history_details["median"]:.2f}m, deviation {history_details["deviation"]:.2f}m',
            'history_details': history_details,
            'corrected_distance': corrected_distance,
            'original_distance': fused_distance,
            'correction_method': 'history_based'
        }
        
        return corrected_distance, corrected_details, {
            "history_correction_applied": True,
            "original_distance": fused_distance,
            "corrected_distance": corrected_distance,
            "original_cluster": majority_vote_details.get('selected_cluster'),
            "corrected_cluster": None,
            "correction_method": "history_based",
            "details": history_details
        }
    
    def process_multi_sensor_fusion(self, sample_token, instance_token, 
                                    lidar_file_path, radar_file_path, camera_file_path):
        """
        Main processing function for multi-sensor fusion with history awareness
        """
        
        results = {
            'sample_token': sample_token,
            'instance_token': instance_token,
            'sensors': {},
            'fused_distance': None,
            'details': {},
            'match_info': {},
            'fusion_details': {},
            'outlier_info': {},
            'majority_vote_details': {},
            'triple_fusion_records': {},  # Store triple fusion record details
            'history_correction': {}  # Store history correction details
        }
        
        print("\n" + "="*50)
        print("Step 1: Loading LIDAR data...")
        print("="*50)
        
        try:
            with open(lidar_file_path, 'r') as f:
                lidar_data = json.load(f)
            print(f"Loaded LIDAR data with {len(lidar_data) if isinstance(lidar_data, list) else 'dict'} records")
        except Exception as e:
            print(f"Error loading LIDAR data: {e}")
            return results
        
        lidar_record = None
        lidar_sample_token = None
        
        if isinstance(lidar_data, list):
            for record in lidar_data:
                if isinstance(record, dict) and record.get('sample_token') == sample_token:
                    lidar_record = record
                    lidar_sample_token = sample_token
                    break
        elif isinstance(lidar_data, dict):
            lidar_record = lidar_data.get(sample_token)
            lidar_sample_token = sample_token
        
        if not lidar_record:
            print(f"No LIDAR record found for sample_token: {sample_token}")
            return results
        
        lidar_distance = None
        if 'Distance' in lidar_record:
            lidar_distance = get_distance_for_instance(lidar_record['Distance'], instance_token)
        
        if lidar_distance is None:
            print(f"No LIDAR distance found for instance: {instance_token}")
            return results
        
        lidar_timestamp = lidar_record.get('timestamp')
        lidar_faulty = lidar_record.get('is_faulty', False)
        
        results['match_info']['lidar'] = {
            'sample_token': lidar_sample_token,
            'timestamp': lidar_timestamp,
            'match_type': 'exact_sample_token',
            'status': 'SUCCESS'
        }
        
        print(f"Found LIDAR record:")
        print(f"  - Sample Token: {lidar_sample_token}")
        print(f"  - Timestamp: {lidar_timestamp}")
        print(f"  - Distance: {lidar_distance:.2f}m")
        print(f"  - Faulty: {lidar_faulty}")
        
        sensor_data = {
            'lidar': {
                'distance': float(lidar_distance),
                'weight': float(self.fusion_system.base_weights['lidar']),
                'available': True,
                'faulty': bool(lidar_faulty),  # Convert to Python bool
                'timestamp': lidar_timestamp,
                'record': lidar_record,
                'sample_token': lidar_sample_token,
                'fusion_type': 'single'
            }
        }
        
        print("\n" + "="*50)
        print("Step 2: Loading RADAR data...")
        print("="*50)
        
        try:
            with open(radar_file_path, 'r') as f:
                radar_data = json.load(f)
            print(f"Loaded RADAR data with {len(radar_data) if isinstance(radar_data, list) else 'dict'} records")
        except Exception as e:
            print(f"Error loading RADAR data: {e}")
            radar_data = []
        
        radar_types = [
            ('radar_front', 'RADAR_FRONT'),
            ('radar_front_right', 'RADAR_FRONT_RIGHT'),
            ('radar_front_left', 'RADAR_FRONT_LEFT')
        ]
        
        for sensor_key, sensor_name in radar_types:
            if isinstance(radar_data, list):
                before_rec, closest_rec, after_rec, status_msg, all_records_info = self.find_three_closest_records(
                    radar_data, lidar_timestamp, sensor_name, instance_token
                )
                
                if closest_rec:
                    print(f"\nProcessing {sensor_name} (triple fusion with Majority Vote):")
                    print(f"  - Status: {status_msg}")
                    
                    # Store record info for JSON output
                    results['triple_fusion_records'][sensor_key] = all_records_info
                    
                    records_to_fuse = []
                    record_info = []
                    
                    if before_rec:
                        records_to_fuse.append(before_rec)
                        record_info.append(f"Before: ts={before_rec['timestamp']}, dist={before_rec['distance']:.2f}m, faulty={before_rec['faulty']}")
                    
                    records_to_fuse.append(closest_rec)
                    record_info.append(f"Closest: ts={closest_rec['timestamp']}, dist={closest_rec['distance']:.2f}m, faulty={closest_rec['faulty']}")
                    
                    if after_rec:
                        records_to_fuse.append(after_rec)
                        record_info.append(f"After: ts={after_rec['timestamp']}, dist={after_rec['distance']:.2f}m, faulty={after_rec['faulty']}")
                    
                    for info in record_info:
                        print(f"    {info}")
                    
                    # Use Majority Vote to fuse three records
                    fused_distance, is_faulty, avg_timestamp, triple_fusion_details = self.fusion_system.fuse_three_records(
                        records_to_fuse, sensor_name
                    )
                    
                    if fused_distance is not None:
                        base_weight = self.fusion_system.base_weights[sensor_key]
                        
                        adjusted_weight = self.fusion_system.calculate_adjusted_weight(
                            base_weight, fused_distance, lidar_distance, is_faulty
                        )
                        
                        sensor_data[sensor_key] = {
                            'distance': float(fused_distance),
                            'weight': float(adjusted_weight),
                            'available': True,
                            'faulty': bool(is_faulty),  # Convert to Python bool
                            'timestamp': avg_timestamp,
                            'time_diff': abs(avg_timestamp - lidar_timestamp) if avg_timestamp else None,
                            'record': closest_rec['record'],
                            'sample_token': closest_rec['record'].get('sample_token'),
                            'fusion_type': 'triple_majority_vote',
                            'num_records_fused': len(records_to_fuse),
                            'triple_fusion_details': triple_fusion_details
                        }
                        
                        # Store fault determination details
                        fault_determination = triple_fusion_details.get('fault_determination', {})
                        results['fusion_details'][sensor_key] = {
                            'num_records': len(records_to_fuse),
                            'records_used': record_info,
                            'fused_distance': float(fused_distance),
                            'result_faulty': bool(is_faulty),  # Convert to Python bool
                            'fusion_method': triple_fusion_details.get('method', 'unknown'),
                            'record_metadata': triple_fusion_details.get('record_metadata', []),
                            'cluster_confidence': float(triple_fusion_details.get('selected_cluster_confidence', 0)),
                            'fault_determination_method': fault_determination.get('method', 'unknown'),
                            'selected_cluster_faulty_count': triple_fusion_details.get('selected_cluster_faulty_count', 0),
                            'selected_cluster_healthy_count': triple_fusion_details.get('selected_cluster_healthy_count', 0)
                        }
                        
                        results['match_info'][sensor_key] = {
                            'sample_token': closest_rec['record'].get('sample_token'),
                            'timestamp': avg_timestamp,
                            'match_type': 'triple_fusion_majority_vote',
                            'status': 'SUCCESS',
                            'num_records_fused': len(records_to_fuse),
                            'details': status_msg,
                            'fusion_method': triple_fusion_details.get('method', 'unknown')
                        }
                        
                        print(f"  - Fused Distance: {fused_distance:.2f}m")
                        print(f"  - Fused Weight: {adjusted_weight:.3f}")
                        print(f"  - Result Faulty: {is_faulty} (determined by selected cluster majority)")
                        
                        # Display decision details
                        if 'fault_determination' in triple_fusion_details:
                            fd = triple_fusion_details['fault_determination']
                            print(f"  - Fault Determination: {fd.get('method', 'unknown')}")
                            if 'selected_cluster_faulty_count' in fd:
                                print(f"  - Selected Cluster: Faulty={fd['selected_cluster_faulty_count']}, Healthy={fd['selected_cluster_healthy_count']}")
                                print(f"  - Decision: Faulty count > Healthy count = {fd.get('is_faulty', False)}")
                        
                        print(f"  - Records Fused: {len(records_to_fuse)}")
                        print(f"  - Fusion Method: {triple_fusion_details.get('method', 'unknown')}")
                        if 'selected_cluster_confidence' in triple_fusion_details:
                            print(f"  - Cluster Confidence: {triple_fusion_details['selected_cluster_confidence']:.2f}")
                    else:
                        print(f"{sensor_name}: Fusion failed")
                        sensor_data[sensor_key] = {'available': False}
                        results['match_info'][sensor_key] = {
                            'status': 'FAILED',
                            'details': 'Fusion of three records failed',
                            'error': status_msg
                        }
                else:
                    print(f"{sensor_name}: {status_msg}")
                    sensor_data[sensor_key] = {'available': False}
                    results['match_info'][sensor_key] = {
                        'status': 'FAILED',
                        'details': status_msg
                    }
            else:
                print(f"{sensor_name}: Radar data is not in list format")
                sensor_data[sensor_key] = {'available': False}
                results['match_info'][sensor_key] = {
                    'status': 'FAILED',
                    'details': 'Radar data format not supported'
                }
        
        print("\n" + "="*50)
        print("Step 3: Loading CAMERA data...")
        print("="*50)
        
        try:
            with open(camera_file_path, 'r') as f:
                camera_data = json.load(f)
            print(f"Loaded CAMERA data with {len(camera_data)} records")
        except Exception as e:
            print(f"Error loading CAMERA data: {e}")
            camera_data = {}
        
        camera_types = [
            ('cam_front', 'CAM_FRONT'),
            ('cam_front_right', 'CAM_FRONT_RIGHT'),
            ('cam_front_left', 'CAM_FRONT_LEFT')
        ]
        
        for sensor_key, sensor_name in camera_types:
            before_rec, closest_rec, after_rec, status_msg, all_records_info = self.find_three_camera_records(
                camera_data, lidar_timestamp, sensor_name, instance_token
            )
            
            if closest_rec:
                print(f"\nProcessing {sensor_name} (triple fusion with Majority Vote):")
                print(f"  - Status: {status_msg}")
                
                # Store record info for JSON output
                results['triple_fusion_records'][sensor_key] = all_records_info
                
                records_to_fuse = []
                record_info = []
                
                if before_rec:
                    records_to_fuse.append(before_rec)
                    record_info.append(f"Before: ts={before_rec['timestamp']}, dist={before_rec['distance']:.2f}m, faulty={before_rec['faulty']}")
                
                records_to_fuse.append(closest_rec)
                record_info.append(f"Closest: ts={closest_rec['timestamp']}, dist={closest_rec['distance']:.2f}m, faulty={closest_rec['faulty']}")
                
                if after_rec:
                    records_to_fuse.append(after_rec)
                    record_info.append(f"After: ts={after_rec['timestamp']}, dist={after_rec['distance']:.2f}m, faulty={after_rec['faulty']}")
                
                for info in record_info:
                    print(f"    {info}")
                
                # Use Majority Vote to fuse three records
                fused_distance, is_faulty, avg_timestamp, triple_fusion_details = self.fusion_system.fuse_three_records(
                    records_to_fuse, sensor_name
                )
                
                if fused_distance is not None:
                    base_weight = self.fusion_system.base_weights[sensor_key]
                    
                    adjusted_weight = self.fusion_system.calculate_adjusted_weight(
                        base_weight, fused_distance, lidar_distance, is_faulty
                    )
                    
                    sensor_data[sensor_key] = {
                        'distance': float(fused_distance),
                        'weight': float(adjusted_weight),
                        'available': True,
                        'faulty': bool(is_faulty),  # Convert to Python bool
                        'timestamp': avg_timestamp,
                        'time_diff': abs(avg_timestamp - lidar_timestamp) if avg_timestamp else None,
                        'record': closest_rec['record'],
                        'sample_token': closest_rec['record_key'],
                        'fusion_type': 'triple_majority_vote',
                        'num_records_fused': len(records_to_fuse),
                        'triple_fusion_details': triple_fusion_details
                    }
                    
                    # Store fault determination details
                    fault_determination = triple_fusion_details.get('fault_determination', {})
                    results['fusion_details'][sensor_key] = {
                        'num_records': len(records_to_fuse),
                        'records_used': record_info,
                        'fused_distance': float(fused_distance),
                        'result_faulty': bool(is_faulty),  # Convert to Python bool
                        'fusion_method': triple_fusion_details.get('method', 'unknown'),
                        'record_metadata': triple_fusion_details.get('record_metadata', []),
                        'cluster_confidence': float(triple_fusion_details.get('selected_cluster_confidence', 0)),
                        'fault_determination_method': fault_determination.get('method', 'unknown'),
                        'selected_cluster_faulty_count': triple_fusion_details.get('selected_cluster_faulty_count', 0),
                        'selected_cluster_healthy_count': triple_fusion_details.get('selected_cluster_healthy_count', 0)
                    }
                    
                    results['match_info'][sensor_key] = {
                        'sample_token': closest_rec['record_key'],
                        'timestamp': avg_timestamp,
                        'match_type': 'triple_fusion_majority_vote',
                        'status': 'SUCCESS',
                        'num_records_fused': len(records_to_fuse),
                        'details': status_msg,
                        'fusion_method': triple_fusion_details.get('method', 'unknown')
                    }
                    
                    print(f"  - Fused Distance: {fused_distance:.2f}m")
                    print(f"  - Fused Weight: {adjusted_weight:.3f}")
                    print(f"  - Result Faulty: {is_faulty} (determined by selected cluster majority)")
                    
                    # Display decision details
                    if 'fault_determination' in triple_fusion_details:
                        fd = triple_fusion_details['fault_determination']
                        print(f"  - Fault Determination: {fd.get('method', 'unknown')}")
                        if 'selected_cluster_faulty_count' in fd:
                            print(f"  - Selected Cluster: Faulty={fd['selected_cluster_faulty_count']}, Healthy={fd['selected_cluster_healthy_count']}")
                            print(f"  - Decision: Faulty count > Healthy count = {fd.get('is_faulty', False)}")
                    
                    print(f"  - Records Fused: {len(records_to_fuse)}")
                    print(f"  - Fusion Method: {triple_fusion_details.get('method', 'unknown')}")
                    if 'selected_cluster_confidence' in triple_fusion_details:
                        print(f"  - Cluster Confidence: {triple_fusion_details['selected_cluster_confidence']:.2f}")
                else:
                    print(f"{sensor_name}: Fusion failed")
                    sensor_data[sensor_key] = {'available': False}
                    results['match_info'][sensor_key] = {
                        'status': 'FAILED',
                        'details': 'Fusion of three records failed',
                        'error': status_msg
                    }
            else:
                print(f"{sensor_name}: {status_msg}")
                sensor_data[sensor_key] = {'available': False}
                results['match_info'][sensor_key] = {
                    'status': 'FAILED',
                    'details': status_msg
                }
        
        print("\n" + "="*50)
        print("Step 4: Outlier detection and weight adjustment...")
        print("="*50)
        
        # Detect outliers and adjust their weights (instead of removing them)
        filtered_sensor_data, outlier_info = self.fusion_system.detect_and_adjust_outliers(sensor_data)
        
        results['outlier_info'] = outlier_info
        
        sensors_with_outliers = list(outlier_info.keys())
        if sensors_with_outliers:
            print(f"\nSensors detected as outliers (weight reduced to {self.fusion_system.outlier_weight_penalty*100:.0f}%):")
            for sensor in sensors_with_outliers:
                if sensor in filtered_sensor_data:
                    print(f"  - {sensor.upper()}: {filtered_sensor_data[sensor]['distance']:.2f}m")
                    print(f"    Original weight: {outlier_info[sensor]['original_weight']:.3f}")
                    print(f"    Adjusted weight: {outlier_info[sensor]['adjusted_weight']:.3f}")
        else:
            print(f"No outliers detected")
        
        print("\n" + "="*50)
        print("Step 5: Performing final multi-sensor fusion using Majority Vote...")
        print("="*50)
        
        fusion_input = {}
        sensor_status = []  # Track sensor status for reporting
        
        for sensor_key in ['lidar', 'radar_front', 'radar_front_right', 'radar_front_left',
                          'cam_front', 'cam_front_right', 'cam_front_left']:
            if (sensor_key in filtered_sensor_data and 
                filtered_sensor_data[sensor_key].get('available')):
                
                # Apply outlier weight factor if sensor is outlier
                if filtered_sensor_data[sensor_key].get('is_outlier', False):
                    outlier_factor = filtered_sensor_data[sensor_key].get('outlier_weight_factor', 
                                                                         self.fusion_system.outlier_weight_penalty)
                    final_weight = filtered_sensor_data[sensor_key]['weight'] * outlier_factor
                    status = 'outlier'
                else:
                    final_weight = filtered_sensor_data[sensor_key]['weight']
                    status = 'normal'
                
                fusion_input[sensor_key] = {
                    'distance': float(filtered_sensor_data[sensor_key]['distance']),
                    'weight': float(final_weight),
                    'available': True,
                    'faulty': bool(filtered_sensor_data[sensor_key]['faulty']),  # Convert to Python bool
                    'is_outlier': bool(filtered_sensor_data[sensor_key].get('is_outlier', False))  # Convert to Python bool
                }
                
                sensor_status.append((sensor_key, status))
            else:
                fusion_input[sensor_key] = {'available': False}
                sensor_status.append((sensor_key, 'not_available'))
        
        if sensor_status:
            print(f"\nSensor status for final fusion:")
            for sensor, status in sensor_status:
                if status == 'outlier':
                    print(f"  - {sensor.upper()}: Included with reduced weight")
                elif status == 'normal':
                    print(f"  - {sensor.upper()}: Included with normal weight")
                else:
                    print(f"  - {sensor.upper()}: Not available")
        
        # Perform fusion with history awareness
        fused_distance, majority_vote_details = self.fusion_system.fuse_all_sensors(
            fusion_input, instance_token
        )
        
        # Apply history-based correction
        if fused_distance is not None:
            corrected_distance, corrected_details, correction_info = self._apply_history_correction(
                instance_token, fused_distance, majority_vote_details
            )
            
            # Update results with corrected values
            fused_distance = corrected_distance
            majority_vote_details = corrected_details
            results['history_correction'] = correction_info
            
            # Update processing history (with corrected distance, not original)
            if instance_token not in self.processing_history:
                self.processing_history[instance_token] = []
            self.processing_history[instance_token].append(float(fused_distance))
            
            # Keep only the last 20 frames in history
            if len(self.processing_history[instance_token]) > 20:
                self.processing_history[instance_token] = self.processing_history[instance_token][-20:]
            
            print(f"\nMAJORITY VOTE FUSION RESULTS (with history awareness):")
            print(f"  Fused distance: {fused_distance:.2f}m")
            print(f"  Fusion method: {majority_vote_details.get('method', 'unknown')}")
            
            if correction_info.get('history_correction_applied', False):
                print(f"  [HISTORY CORRECTION APPLIED]")
                print(f"    Original: {correction_info.get('original_distance', 0):.2f}m")
                print(f"    Corrected: {correction_info.get('corrected_distance', 0):.2f}m")
                print(f"    Correction method: {correction_info.get('correction_method', 'N/A')}")
                if 'original_cluster' in correction_info and correction_info['original_cluster'] is not None:
                    print(f"    Original cluster: {correction_info['original_cluster']}")
                if 'corrected_cluster' in correction_info and correction_info['corrected_cluster'] is not None:
                    print(f"    Corrected cluster: {correction_info['corrected_cluster']}")
            
            if majority_vote_details.get('method') == 'majority_vote':
                print(f"  Clusters identified: {majority_vote_details.get('clusters_count', 0)}")
                print(f"  Selected cluster: {majority_vote_details.get('selected_cluster', 'N/A')}")
                print(f"  Selected cluster confidence: {majority_vote_details.get('selected_cluster_confidence', 0):.2f}")
                
                cluster_analysis = majority_vote_details.get('cluster_analysis', [])
                for i, analysis in enumerate(cluster_analysis):
                    if i == majority_vote_details.get('selected_cluster'):
                        marker = " [SELECTED]"
                    else:
                        marker = ""
                    print(f"  Cluster {i}{marker}: {analysis['sensor_count']} sensors, "
                          f"Health ratio: {analysis['health_ratio']:.2f}, "
                          f"Mean distance: {analysis['mean_distance']:.2f}m")
                    if 'confidence' in analysis:
                        print(f"    Confidence: {analysis['confidence']:.2f}")
            
            print("\n  Sensor contributions (with outlier weight adjustment):")
            for sensor_key in ['lidar', 'radar_front', 'radar_front_right', 'radar_front_left',
                              'cam_front', 'cam_front_right', 'cam_front_left']:
                if fusion_input[sensor_key]['available']:
                    data = fusion_input[sensor_key]
                    outlier_mark = " [OUTLIER]" if data.get('is_outlier', False) else ""
                    print(f"    {sensor_key}{outlier_mark}: distance={data['distance']:.2f}m, "
                          f"weight={data['weight']:.3f}, faulty={data['faulty']}")
        else:
            print(f"Final fusion failed - no available sensors")
        
        results['sensors'] = filtered_sensor_data
        results['fused_distance'] = float(fused_distance) if fused_distance is not None else None
        results['majority_vote_details'] = majority_vote_details
        
        available_sensors = [s for s in filtered_sensor_data if 
                           filtered_sensor_data[s].get('available')]
        results['details']['available_sensors'] = available_sensors
        results['details']['num_sensors'] = len(available_sensors)
        results['details']['sensor_status'] = sensor_status
        
        return results
    
    def display_results(self, results):
        """Display formatted results with detailed match and outlier information"""
        print("\n" + "="*80)
        print("MULTI-SENSOR MAJORITY VOTE FUSION RESULTS (WITH HISTORY AWARENESS)")
        print("="*80)
        
        print(f"\nSample Token: {results['sample_token']}")
        print(f"Instance Token: {results['instance_token']}")
        
        # Display history correction details
        if results.get('history_correction', {}).get('history_correction_applied', False):
            print("\n" + "-"*80)
            print("HISTORY-BASED CORRECTION APPLIED:")
            print("-"*80)
            
            correction = results['history_correction']
            print(f"Original distance: {correction.get('original_distance', 0):.2f}m")
            print(f"Corrected distance: {correction.get('corrected_distance', 0):.2f}m")
            print(f"Correction method: {correction.get('correction_method', 'N/A')}")
            if 'original_cluster' in correction and correction['original_cluster'] is not None:
                print(f"Original cluster: {correction['original_cluster']}")
            if 'corrected_cluster' in correction and correction['corrected_cluster'] is not None:
                print(f"Corrected cluster: {correction['corrected_cluster']}")
            print(f"Reason: {correction.get('reason', 'N/A')}")
            
            if 'details' in correction:
                details = correction['details']
                print(f"\nHistory analysis:")
                print(f"  History size: {details.get('history_size', 0)} frames")
                print(f"  History median: {details.get('median', 0):.2f}m")
                print(f"  History std: {details.get('std', 0):.2f}m")
                print(f"  Adaptive threshold: {details.get('adaptive_threshold', 0):.2f}m")
                print(f"  Deviation: {details.get('deviation', 0):.2f}m")
                
                # Display trend analysis if available
                if 'threshold_info' in details and 'trend_info' in details['threshold_info']:
                    trend_info = details['threshold_info']['trend_info']
                    print(f"\n  Trend analysis:")
                    print(f"    Has history: {trend_info.get('has_history', False)}")
                    print(f"    Is moving: {trend_info.get('is_moving', False)}")
                    print(f"    Direction: {trend_info.get('movement_direction', 'unknown')}")
                    print(f"    Avg change: {trend_info.get('avg_change', 0):.2f}m/frame")
                    print(f"    Avg abs change: {trend_info.get('avg_abs_change', 0):.2f}m/frame")
                    print(f"    Trend confidence: {trend_info.get('trend_confidence', 0):.2f}")
        
        # Display triple fusion record details
        if results.get('triple_fusion_records'):
            print("\n" + "-"*80)
            print("TRIPLE FUSION RECORD DETAILS:")
            print("-"*80)
            
            for sensor_key, records in results['triple_fusion_records'].items():
                if records:
                    print(f"\n{sensor_key.upper()}:")
                    for record in records:
                        print(f"  {record['position'].upper()}:")
                        print(f"    Sample Token: {record.get('sample_token', 'N/A')}")
                        print(f"    Timestamp: {record.get('timestamp', 'N/A')}")
                        print(f"    Distance: {record.get('distance', 'N/A'):.2f}m")
                        print(f"    Faulty: {record.get('faulty', 'N/A')}")
        
        if results['outlier_info']:
            print("\n" + "-"*80)
            print("OUTLIER DETECTION RESULTS:")
            print("-"*80)
            
            for sensor, info in results['outlier_info'].items():
                print(f"\n{sensor.upper()}:")
                print(f"  Distance: {info['distance']:.2f}m")
                print(f"  Detection methods: {', '.join(info['methods'])}")
                print(f"  Original weight: {info['original_weight']:.3f}")
                print(f"  Adjusted weight: {info['adjusted_weight']:.3f}")
                
                if 'iqr' in info['methods']:
                    details = info['details']
                    print(f"  IQR Analysis:")
                    print(f"    - Lower bound: {details['lower_bound']:.2f}m")
                    print(f"    - Upper bound: {details['upper_bound']:.2f}m")
                    print(f"    - Deviation from median: {details['deviation_percent']:.1f}%")
                
                if 'mad' in info['methods']:
                    details = info['details']
                    mod_z = details.get('modified_z_score')
                    threshold = details.get('threshold')
                    median = details.get('median')
                    if mod_z is not None:
                        print(f"  MAD Analysis:")
                        print(f"    - Modified Z-score: {mod_z:.2f}")
                        print(f"    - Threshold: {threshold}")
                        print(f"    - Median: {median:.2f}m")
                    else:
                        print(f"  MAD Analysis: N/A")
        
        print("\n" + "-"*80)
        print("MATCH INFORMATION:")
        print("-"*80)
        
        sensor_order = [
            ('lidar', 'LIDAR'),
            ('radar_front', 'RADAR_FRONT'),
            ('radar_front_right', 'RADAR_FRONT_RIGHT'),
            ('radar_front_left', 'RADAR_FRONT_LEFT'),
            ('cam_front', 'CAM_FRONT'),
            ('cam_front_right', 'CAM_FRONT_RIGHT'),
            ('cam_front_left', 'CAM_FRONT_LEFT')
        ]
        
        print(f"{'SENSOR':20} | {'STATUS':15} | {'DETAILS'}")
        print("-" * 80)
        
        for sensor_key, sensor_name in sensor_order:
            if sensor_key in results['match_info']:
                match_info = results['match_info'][sensor_key]
                status = match_info.get('status', 'UNKNOWN')
                details = match_info.get('details', 'No details')
                
                is_outlier = sensor_key in results['outlier_info']
                
                if status == 'SUCCESS':
                    if is_outlier:
                        status_display = "OUTLIER"
                    else:
                        status_display = "MATCHED"
                elif status == 'FAILED':
                    status_display = "FAILED"
                else:
                    status_display = status
                
                print(f"{sensor_name:20} | {status_display:15} | {details}")
            else:
                print(f"{sensor_name:20} | {'NO INFO':15} | No match information available")
        
        print("\n" + "-"*80)
        print("SENSOR READINGS (With Outlier Weight Adjustment):")
        print("-"*80)
        
        for sensor_key, sensor_name in sensor_order:
            if sensor_key in results['sensors'] and results['sensors'][sensor_key].get('available'):
                data = results['sensors'][sensor_key]
                fusion_type = data.get('fusion_type', 'single')
                num_records = data.get('num_records_fused', 1)
                
                fusion_info = f" ({fusion_type}, {num_records} recs)" if fusion_type.startswith('triple') else ""
                
                is_outlier = data.get('is_outlier', False)
                outlier_factor = data.get('outlier_weight_factor', 1.0)
                
                if is_outlier:
                    outlier_mark = f" [OUTLIER, weight × {outlier_factor:.1f}]"
                else:
                    outlier_mark = ""
                
                print(f"{sensor_name:20} | "
                      f"Distance: {data['distance']:7.2f}m | "
                      f"Weight: {data['weight']:6.3f} | "
                      f"Faulty: {data['faulty']:5} | "
                      f"Type: {fusion_type}{fusion_info}{outlier_mark}")
                
                # Display fault determination details for triple fusion sensors
                if fusion_type.startswith('triple') and 'triple_fusion_details' in data:
                    fd = data['triple_fusion_details'].get('fault_determination', {})
                    if fd:
                        print(f"{'':20} |   Fault Determination: {fd.get('method', 'unknown')}")
                        if 'selected_cluster_faulty_count' in fd:
                            print(f"{'':20} |   Selected Cluster: Faulty={fd['selected_cluster_faulty_count']}, Healthy={fd['selected_cluster_healthy_count']}")
            else:
                if sensor_key in results['match_info']:
                    match_info = results['match_info'][sensor_key]
                    reason = match_info.get('details', 'Unknown')
                    print(f"{sensor_name:20} | Not available - {reason}")
                else:
                    print(f"{sensor_name:20} | Not available")
        
        print("\n" + "-"*80)
        print("MAJORITY VOTE FUSION DETAILS:")
        print("-"*80)
        
        if results['fused_distance'] is not None:
            print(f"Final Fused Distance: {results['fused_distance']:.2f}m")
            print(f"Number of sensors used: {results['details']['num_sensors']}")
            
            # Count outliers included in fusion
            outlier_count = sum(1 for s, status in results['details'].get('sensor_status', []) 
                              if status == 'outlier')
            if outlier_count > 0:
                print(f"Sensors with reduced weight (outliers): {outlier_count}")
            
            majority_details = results.get('majority_vote_details', {})
            method = majority_details.get('method', 'unknown')
            print(f"Fusion method: {method}")
            
            if 'history_used' in majority_details and majority_details['history_used']:
                print(f"History used: Yes ({majority_details.get('history_size', 0)} previous frames)")
            
            if method == 'majority_vote':
                print(f"\nCluster Analysis:")
                clusters = majority_details.get('cluster_analysis', [])
                for i, cluster in enumerate(clusters):
                    if i == majority_details.get('selected_cluster'):
                        marker = " [SELECTED]"
                    else:
                        marker = ""
                    print(f"  Cluster {i}{marker}:")
                    print(f"    Sensors: {', '.join(cluster['sensors'])}")
                    print(f"    Count: {cluster['sensor_count']} (Healthy: {cluster['healthy_count']}, Faulty: {cluster['faulty_count']})")
                    print(f"    Health Ratio: {cluster['health_ratio']:.2f}")
                    print(f"    Mean Distance: {cluster['mean_distance']:.2f}m ± {cluster['std_distance']:.2f}m")
                    if 'confidence' in cluster:
                        print(f"    Confidence: {cluster['confidence']:.2f}")
            
            triple_fusion_sensors = []
            for sensor_key in results['sensors']:
                if (results['sensors'][sensor_key].get('available') and 
                    results['sensors'][sensor_key].get('fusion_type', '').startswith('triple')):
                    triple_fusion_sensors.append(sensor_key)
            
            if triple_fusion_sensors:
                print(f"Sensors using triple fusion with Majority Vote: {', '.join(triple_fusion_sensors)}")
        else:
            print("No fused distance available - insufficient sensor data")
        
        print("="*80)

# ----------------------------
# FUNCTION TO READ TOKENS FROM TXT FILE
# ----------------------------

def read_tokens_from_txt(file_path):
    """
    Read instance_token and sample_tokens from a TXT file.
    Expected format:
    Instance-token:
    c542c70024df4cf2a02c55a7b16fe3c4
    Sample-token:
    interpolated_67e5f889_29e056fc_190b427f
    interpolated_67e5f889_29e056fc_f95ca694
    ...
    """
    instance_token = None
    sample_tokens = []
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Clean lines
        lines = [line.strip() for line in lines if line.strip()]
        
        i = 0
        while i < len(lines):
            if lines[i].lower().startswith('instance-token'):
                i += 1
                if i < len(lines):
                    instance_token = lines[i]
                    i += 1
            elif lines[i].lower().startswith('sample-token'):
                i += 1
                while i < len(lines) and not lines[i].lower().startswith(('instance-token', 'sample-token')):
                    if lines[i] and not lines[i].startswith('#'):  # Skip comments
                        sample_tokens.append(lines[i])
                    i += 1
            else:
                i += 1
        
        print(f"\nRead from file: Instance token = {instance_token}")
        print(f"Number of sample tokens: {len(sample_tokens)}")
        
        if not instance_token:
            print("ERROR: No instance token found in the file!")
            return None, []
        
        if not sample_tokens:
            print("ERROR: No sample tokens found in the file!")
            return None, []
        
        return instance_token, sample_tokens
        
    except Exception as e:
        print(f"Error reading tokens file: {e}")
        return None, []

# ----------------------------
# BATCH PROCESSING FUNCTION
# ----------------------------

def batch_process_tokens_file(tokens_file_path, lidar_file, radar_file, camera_file, output_dir):
    """
    Process multiple sample tokens from a TXT file
    """
    # Read tokens from file
    instance_token, sample_tokens = read_tokens_from_txt(tokens_file_path)
    
    if not instance_token or not sample_tokens:
        print("Failed to read tokens from file. Exiting.")
        return
    
    print(f"\n{'='*80}")
    print(f"STARTING BATCH PROCESSING")
    print(f"Instance Token: {instance_token}")
    print(f"Number of Sample Tokens: {len(sample_tokens)}")
    print(f"{'='*80}")
    
    # Create main output directory
    main_output_dir = os.path.join(output_dir, f"batch_{instance_token[:8]}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Create subdirectories for JSON and TXT outputs
    json_output_dir = os.path.join(main_output_dir, "JSON")
    txt_output_dir = os.path.join(main_output_dir, "TXT")
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(txt_output_dir, exist_ok=True)
    
    # Create summary file
    summary_file_path = os.path.join(main_output_dir, f"batch_summary_{instance_token[:8]}.txt")
    summary_data = []
    
    processor = MultiSensorProcessor()
    
    # Process each sample token
    for idx, sample_token in enumerate(sample_tokens, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING {idx}/{len(sample_tokens)}")
        print(f"Sample Token: {sample_token}")
        print(f"{'='*80}")
        
        # Process the sample
        results = processor.process_multi_sensor_fusion(
            sample_token=sample_token,
            instance_token=instance_token,
            lidar_file_path=lidar_file,
            radar_file_path=radar_file,
            camera_file_path=camera_file
        )
        
        # Display results
        processor.display_results(results)
        
        # Prepare output filename (clean illegal characters)
        safe_sample_token = sample_token.replace('\\', '_').replace('/', '_').replace(':', '_')
        output_filename_base = f"multi_sensor_fusion_{safe_sample_token}_{instance_token[:8]}"
        
        # Save JSON output
        json_output_path = os.path.join(json_output_dir, f"{output_filename_base}.json")
        
        # Prepare serializable results
        serializable_results = {
            'sample_token': results['sample_token'],
            'instance_token': results['instance_token'],
            'fused_distance': float(results['fused_distance']) if results['fused_distance'] is not None else None,
            'details': results['details'],
            'match_info': results['match_info'],
            'fusion_details': results.get('fusion_details', {}),
            'outlier_info': results.get('outlier_info', {}),
            'majority_vote_details': results.get('majority_vote_details', {}),
            'triple_fusion_records': results.get('triple_fusion_records', {}),
            'history_correction': results.get('history_correction', {}),
            'sensors': {}
        }
        
        # Convert sensors data to serializable format
        for sensor_key, sensor_data in results['sensors'].items():
            if isinstance(sensor_data, dict) and sensor_data.get('available'):
                sensor_entry = {
                    'distance': float(sensor_data.get('distance')) if sensor_data.get('distance') is not None else None,
                    'weight': float(sensor_data.get('weight')) if sensor_data.get('weight') is not None else None,
                    'faulty': bool(sensor_data.get('faulty')) if sensor_data.get('faulty') is not None else False,
                    'timestamp': sensor_data.get('timestamp'),
                    'time_diff': sensor_data.get('time_diff'),
                    'sample_token': sensor_data.get('sample_token'),
                    'fusion_type': sensor_data.get('fusion_type', 'single'),
                    'num_records_fused': sensor_data.get('num_records_fused', 1),
                    'is_outlier': bool(sensor_data.get('is_outlier', False)),
                    'outlier_weight_factor': float(sensor_data.get('outlier_weight_factor', 1.0))
                }
                
                # Add triple fusion details if available (ensure all bools are converted)
                if 'triple_fusion_details' in sensor_data:
                    triple_details = sensor_data['triple_fusion_details'].copy()
                    # Convert any numpy bools in triple details
                    if 'fault_determination' in triple_details:
                        fd = triple_details['fault_determination']
                        if 'is_faulty' in fd:
                            fd['is_faulty'] = bool(fd['is_faulty'])
                    sensor_entry['triple_fusion_details'] = triple_details
                
                serializable_results['sensors'][sensor_key] = sensor_entry
            else:
                serializable_results['sensors'][sensor_key] = {'available': False}
        
        # Save JSON using custom encoder
        with open(json_output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"\nJSON results saved to: {json_output_path}")
        
        # Save TXT output
        txt_output_path = os.path.join(txt_output_dir, f"{output_filename_base}.txt")
        
        with open(txt_output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-SENSOR MAJORITY VOTE FUSION RESULTS (WITH HISTORY AWARENESS)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Sample Token: {sample_token}\n")
            f.write(f"Instance Token: {instance_token}\n\n")
            
            # Write history correction details
            if results.get('history_correction', {}).get('history_correction_applied', False):
                f.write("-"*80 + "\n")
                f.write("HISTORY-BASED CORRECTION:\n")
                f.write("-"*80 + "\n\n")
                
                correction = results['history_correction']
                f.write(f"Original distance: {correction.get('original_distance', 0):.2f}m\n")
                f.write(f"Corrected distance: {correction.get('corrected_distance', 0):.2f}m\n")
                f.write(f"Correction method: {correction.get('correction_method', 'N/A')}\n")
                if 'original_cluster' in correction and correction['original_cluster'] is not None:
                    f.write(f"Original cluster: {correction['original_cluster']}\n")
                if 'corrected_cluster' in correction and correction['corrected_cluster'] is not None:
                    f.write(f"Corrected cluster: {correction['corrected_cluster']}\n")
                f.write(f"Reason: {correction.get('reason', 'N/A')}\n")
                
                if 'details' in correction:
                    details = correction['details']
                    f.write(f"\nHistory analysis:\n")
                    f.write(f"  History size: {details.get('history_size', 0)} frames\n")
                    f.write(f"  History median: {details.get('median', 0):.2f}m\n")
                    f.write(f"  History std: {details.get('std', 0):.2f}m\n")
                    f.write(f"  Adaptive threshold: {details.get('adaptive_threshold', 0):.2f}m\n")
                    f.write(f"  Deviation: {details.get('deviation', 0):.2f}m\n")
                    
                    # Write trend analysis if available
                    if 'threshold_info' in details and 'trend_info' in details['threshold_info']:
                        trend_info = details['threshold_info']['trend_info']
                        f.write(f"\n  Trend analysis:\n")
                        f.write(f"    Has history: {trend_info.get('has_history', False)}\n")
                        f.write(f"    Is moving: {trend_info.get('is_moving', False)}\n")
                        f.write(f"    Direction: {trend_info.get('movement_direction', 'unknown')}\n")
                        f.write(f"    Avg change: {trend_info.get('avg_change', 0):.2f}m/frame\n")
                        f.write(f"    Avg abs change: {trend_info.get('avg_abs_change', 0):.2f}m/frame\n")
                        f.write(f"    Trend confidence: {trend_info.get('trend_confidence', 0):.2f}\n")
                f.write("\n")
            
            # Write triple fusion record details
            if results.get('triple_fusion_records'):
                f.write("-"*80 + "\n")
                f.write("TRIPLE FUSION RECORD DETAILS:\n")
                f.write("-"*80 + "\n\n")
                
                for sensor_key, records in results['triple_fusion_records'].items():
                    if records:
                        f.write(f"{sensor_key.upper()}:\n")
                        for record in records:
                            f.write(f"  {record['position'].upper()}:\n")
                            f.write(f"    Sample Token: {record.get('sample_token', 'N/A')}\n")
                            f.write(f"    Timestamp: {record.get('timestamp', 'N/A')}\n")
                            f.write(f"    Distance: {record.get('distance', 'N/A'):.2f}m\n")
                            f.write(f"    Faulty: {record.get('faulty', 'N/A')}\n")
                        f.write("\n")
            
            if results['outlier_info']:
                f.write("-"*80 + "\n")
                f.write("OUTLIER DETECTION:\n")
                f.write("-"*80 + "\n\n")
                
                for sensor, info in results['outlier_info'].items():
                    f.write(f"{sensor.upper()}:\n")
                    f.write(f"  Distance: {info['distance']:.2f}m\n")
                    f.write(f"  Detection methods: {', '.join(info['methods'])}\n")
                    f.write(f"  Original weight: {info['original_weight']:.3f}\n")
                    f.write(f"  Adjusted weight: {info['adjusted_weight']:.3f}\n")
                    
                    if 'iqr' in info['methods']:
                        details = info['details']
                        f.write(f"  IQR Analysis:\n")
                        f.write(f"    - Bounds: [{details['lower_bound']:.2f}, {details['upper_bound']:.2f}]\n")
                        f.write(f"    - Deviation from median: {details['deviation_percent']:.1f}%\n")
                    
                    if 'mad' in info['methods']:
                        details = info['details']
                        mod_z = details.get('modified_z_score')
                        threshold = details.get('threshold')
                        if mod_z is not None:
                            f.write(f"  MAD Analysis:\n")
                            f.write(f"    - Modified Z-score: {mod_z:.2f}\n")
                            f.write(f"    - Threshold: {threshold}\n")
                        else:
                            f.write(f"  MAD Analysis: N/A\n")
                    f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("MATCH INFORMATION:\n")
            f.write("-"*80 + "\n")
            
            sensor_order = [
                ('lidar', 'LIDAR'),
                ('radar_front', 'RADAR_FRONT'),
                ('radar_front_right', 'RADAR_FRONT_RIGHT'),
                ('radar_front_left', 'RADAR_FRONT_LEFT'),
                ('cam_front', 'CAM_FRONT'),
                ('cam_front_right', 'CAM_FRONT_RIGHT'),
                ('cam_front_left', 'CAM_FRONT_LEFT')
            ]
            
            for sensor_key, sensor_name in sensor_order:
                if sensor_key in results['match_info']:
                    match_info = results['match_info'][sensor_key]
                    f.write(f"{sensor_name}:\n")
                    f.write(f"  Status: {match_info.get('status', 'UNKNOWN')}\n")
                    f.write(f"  Details: {match_info.get('details', 'No details')}\n")
                    if 'sample_token' in match_info and match_info['sample_token']:
                        f.write(f"  Sample Token: {match_info['sample_token']}\n")
                    if 'timestamp' in match_info and match_info['timestamp']:
                        f.write(f"  Timestamp: {match_info['timestamp']}\n")
                    if 'num_records_fused' in match_info:
                        f.write(f"  Records Fused: {match_info['num_records_fused']}\n")
                    if 'fusion_method' in match_info:
                        f.write(f"  Fusion Method: {match_info['fusion_method']}\n")
                    f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("SENSOR READINGS:\n")
            f.write("-"*80 + "\n")
            
            for sensor_key, sensor_name in sensor_order:
                if sensor_key in results['sensors'] and results['sensors'][sensor_key].get('available'):
                    data = results['sensors'][sensor_key]
                    fusion_info = f" ({data.get('fusion_type', 'single')}, {data.get('num_records_fused', 1)} records)" if data.get('fusion_type', '').startswith('triple') else ""
                    outlier_info = f" [OUTLIER, weight × {data.get('outlier_weight_factor', 1.0):.1f}]" if data.get('is_outlier', False) else ""
                    
                    f.write(f"{sensor_name:20} | "
                           f"Distance: {data['distance']:7.2f}m | "
                           f"Weight: {data['weight']:6.3f} | "
                           f"Faulty: {data['faulty']:5}{fusion_info}{outlier_info}\n")
                    
                    # Write fault determination details for triple fusion sensors
                    if data.get('fusion_type', '').startswith('triple') and 'triple_fusion_details' in data:
                        fd = data['triple_fusion_details'].get('fault_determination', {})
                        if fd:
                            f.write(f"{'':20} |   Fault Determination: {fd.get('method', 'unknown')}\n")
                            if 'selected_cluster_faulty_count' in fd:
                                f.write(f"{'':20} |   Selected Cluster: Faulty={fd['selected_cluster_faulty_count']}, Healthy={fd['selected_cluster_healthy_count']}\n")
                else:
                    if sensor_key in results['match_info']:
                        reason = results['match_info'][sensor_key].get('details', 'Unknown')
                        f.write(f"{sensor_name:20} | Not available - {reason}\n")
                    else:
                        f.write(f"{sensor_name:20} | Not available\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("MAJORITY VOTE FUSION RESULTS:\n")
            f.write("-"*80 + "\n")
            
            if results['fused_distance'] is not None:
                f.write(f"Final Fused Distance: {results['fused_distance']:.2f}m\n")
                f.write(f"Number of sensors used: {results['details']['num_sensors']}\n")
                
                # Count outliers
                outlier_count = sum(1 for s, status in results['details'].get('sensor_status', []) 
                                  if status == 'outlier')
                if outlier_count > 0:
                    f.write(f"Sensors with reduced weight (outliers): {outlier_count}\n")
                
                majority_details = results.get('majority_vote_details', {})
                f.write(f"Fusion method: {majority_details.get('method', 'unknown')}\n")
                
                if 'history_used' in majority_details and majority_details['history_used']:
                    f.write(f"History used: Yes ({majority_details.get('history_size', 0)} previous frames)\n")
                
                if majority_details.get('method') == 'majority_vote':
                    f.write(f"\nCluster Analysis:\n")
                    clusters = majority_details.get('cluster_analysis', [])
                    for i, cluster in enumerate(clusters):
                        if i == majority_details.get('selected_cluster'):
                            marker = " [SELECTED]"
                        else:
                            marker = ""
                        f.write(f"  Cluster {i}{marker}:\n")
                        f.write(f"    Sensors: {', '.join(cluster['sensors'])}\n")
                        f.write(f"    Count: {cluster['sensor_count']} (Healthy: {cluster['healthy_count']}, Faulty: {cluster['faulty_count']})\n")
                        f.write(f"    Health Ratio: {cluster['health_ratio']:.2f}\n")
                        f.write(f"    Mean Distance: {cluster['mean_distance']:.2f}m\n")
                        if 'confidence' in cluster:
                            f.write(f"    Confidence: {cluster['confidence']:.2f}\n")
            else:
                f.write("No fused distance available\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"TXT summary saved to: {txt_output_path}")
        
        # Add to summary data
        summary_entry = {
            'sample_token': sample_token,
            'fused_distance': float(results['fused_distance']) if results['fused_distance'] is not None else None,
            'num_sensors': results['details']['num_sensors'],
            'available_sensors': results['details']['available_sensors'],
            'outlier_count': len(results['outlier_info']),
            'history_correction_applied': results.get('history_correction', {}).get('history_correction_applied', False)
        }
        summary_data.append(summary_entry)
    
    # Write summary file
    with open(summary_file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"BATCH PROCESSING SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Instance Token: {instance_token}\n")
        f.write(f"Total Samples Processed: {len(sample_tokens)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("="*80 + "\n")
        f.write("SAMPLE TOKEN RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Sample Token':40} | {'Fused Distance':15} | {'Sensors':8} | {'Outliers':8} | {'History Corr':12}\n")
        f.write("-"*105 + "\n")
        
        for entry in summary_data:
            fused_dist_str = f"{entry['fused_distance']:.2f}m" if entry['fused_distance'] is not None else "N/A"
            history_corr = "Yes" if entry.get('history_correction_applied', False) else "No"
            f.write(f"{entry['sample_token']:40} | {fused_dist_str:15} | {entry['num_sensors']:8} | {entry['outlier_count']:8} | {history_corr:12}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        fused_distances = [e['fused_distance'] for e in summary_data if e['fused_distance'] is not None]
        if fused_distances:
            f.write(f"Fused Distance Statistics:\n")
            f.write(f"  - Min: {min(fused_distances):.2f}m\n")
            f.write(f"  - Max: {max(fused_distances):.2f}m\n")
            f.write(f"  - Mean: {np.mean(fused_distances):.2f}m\n")
            f.write(f"  - Std: {np.std(fused_distances):.2f}m\n\n")
        
        sensor_counts = [e['num_sensors'] for e in summary_data]
        if sensor_counts:
            f.write(f"Sensor Count Statistics:\n")
            f.write(f"  - Min: {min(sensor_counts)}\n")
            f.write(f"  - Max: {max(sensor_counts)}\n")
            f.write(f"  - Mean: {np.mean(sensor_counts):.1f}\n\n")
        
        outlier_counts = [e['outlier_count'] for e in summary_data]
        if outlier_counts:
            f.write(f"Outlier Count Statistics:\n")
            f.write(f"  - Total outliers detected: {sum(outlier_counts)}\n")
            f.write(f"  - Samples with outliers: {sum(1 for c in outlier_counts if c > 0)}/{len(outlier_counts)}\n")
            f.write(f"  - Max outliers in a sample: {max(outlier_counts)}\n\n")
        
        history_corrections = [e.get('history_correction_applied', False) for e in summary_data]
        if history_corrections:
            f.write(f"History Correction Statistics:\n")
            f.write(f"  - Samples corrected by history: {sum(1 for c in history_corrections if c)}/{len(history_corrections)}\n")
            f.write(f"  - Percentage: {sum(1 for c in history_corrections if c)/len(history_corrections)*100:.1f}%\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OUTPUT DIRECTORY STRUCTURE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Main Directory: {main_output_dir}\n")
        f.write(f"JSON Files: {json_output_dir}\n")
        f.write(f"TXT Files: {txt_output_dir}\n")
        f.write(f"Summary File: {summary_file_path}\n")
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE!")
    print(f"Total samples processed: {len(sample_tokens)}")
    print(f"Output directory: {main_output_dir}")
    print(f"Summary file: {summary_file_path}")
    print(f"{'='*80}")

# ----------------------------
# MAIN CONFIGURATION
# ----------------------------

OUTPUT_DIR = r'C:\PATH\AMVF'

LIDAR_FILE = r'C:\PATH\lidar_file.json'
RADAR_FILE = r'C:\PATH\radar_file.json'
CAMERA_FILE = r'C:\PATH\camera_file.json'

# ----------------------------
# MAIN EXECUTION
# ----------------------------

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MULTI-SENSOR MAJORITY VOTE FUSION SYSTEM WITH HISTORY AWARENESS")
    print("BATCH PROCESSING MODE")
    print("="*80)
    
    # Ask user for input method
    print("\nSelect input method:")
    print("1. Use TXT file with multiple sample tokens")
    print("2. Enter single sample token and instance token manually")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Get TXT file path from user
        tokens_file_path = input("\nEnter path to TXT file with tokens: ").strip()
        
        if not tokens_file_path:
            # Default path for testing
            tokens_file_path = r"C:\PATH\tokens.txt"
            print(f"Using default: {tokens_file_path}")
        
        # Check if file exists
        if not os.path.exists(tokens_file_path):
            print(f"ERROR: File not found: {tokens_file_path}")
            print("Please create a TXT file with the following format:")
            print("Instance-token:")
            print("c542c70024df4cf2a02c55a7b16fe3c4")
            print("Sample-token:")
            print("interpolated_67e5f889_29e056fc_190b427f")
            print("interpolated_67e5f889_29e056fc_f95ca694")
            print("...")
        else:
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            
            # Run batch processing
            batch_process_tokens_file(
                tokens_file_path=tokens_file_path,
                lidar_file=LIDAR_FILE,
                radar_file=RADAR_FILE,
                camera_file=CAMERA_FILE,
                output_dir=OUTPUT_DIR
            )
    
    elif choice == "2":
        # Original single processing mode
        sample_token = input("\nEnter sample token: ").strip()
        if not sample_token:
            sample_token = "interpolated_0d0700a2_de7593d7_ffe591af"
            print(f"Using default: {sample_token}")
        
        instance_token = input("Enter instance token: ").strip()
        if not instance_token:
            instance_token = "d0f5aace57684923b8a44b554b006fb2"
            print(f"Using default: {instance_token}")
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        processor = MultiSensorProcessor()
        
        results = processor.process_multi_sensor_fusion(
            sample_token=sample_token,
            instance_token=instance_token,
            lidar_file_path=LIDAR_FILE,
            radar_file_path=RADAR_FILE,
            camera_file_path=CAMERA_FILE
        )
        
        processor.display_results(results)
        
        output_filename = f"multi_sensor_fusion_{sample_token}_{instance_token}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Prepare serializable results
        serializable_results = {
            'sample_token': results['sample_token'],
            'instance_token': results['instance_token'],
            'fused_distance': float(results['fused_distance']) if results['fused_distance'] is not None else None,
            'details': results['details'],
            'match_info': results['match_info'],
            'fusion_details': results.get('fusion_details', {}),
            'outlier_info': results.get('outlier_info', {}),
            'majority_vote_details': results.get('majority_vote_details', {}),
            'triple_fusion_records': results.get('triple_fusion_records', {}),
            'history_correction': results.get('history_correction', {}),
            'sensors': {}
        }
        
        # Convert sensors data to serializable format
        for sensor_key, sensor_data in results['sensors'].items():
            if isinstance(sensor_data, dict) and sensor_data.get('available'):
                sensor_entry = {
                    'distance': float(sensor_data.get('distance')) if sensor_data.get('distance') is not None else None,
                    'weight': float(sensor_data.get('weight')) if sensor_data.get('weight') is not None else None,
                    'faulty': bool(sensor_data.get('faulty')) if sensor_data.get('faulty') is not None else False,
                    'timestamp': sensor_data.get('timestamp'),
                    'time_diff': sensor_data.get('time_diff'),
                    'sample_token': sensor_data.get('sample_token'),
                    'fusion_type': sensor_data.get('fusion_type', 'single'),
                    'num_records_fused': sensor_data.get('num_records_fused', 1),
                    'is_outlier': bool(sensor_data.get('is_outlier', False)),
                    'outlier_weight_factor': float(sensor_data.get('outlier_weight_factor', 1.0))
                }
                
                # Add triple fusion details if available
                if 'triple_fusion_details' in sensor_data:
                    triple_details = sensor_data['triple_fusion_details'].copy()
                    # Convert any numpy bools in triple details
                    if 'fault_determination' in triple_details:
                        fd = triple_details['fault_determination']
                        if 'is_faulty' in fd:
                            fd['is_faulty'] = bool(fd['is_faulty'])
                    sensor_entry['triple_fusion_details'] = triple_details
                
                serializable_results['sensors'][sensor_key] = sensor_entry
            else:
                serializable_results['sensors'][sensor_key] = {'available': False}
        
        # Save JSON using custom encoder
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"\nResults saved to: {output_path}")
        
        txt_filename = f"multi_sensor_fusion_summary_{sample_token}_{instance_token}.txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)
        
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-SENSOR MAJORITY VOTE FUSION RESULTS (WITH HISTORY AWARENESS)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Sample Token: {sample_token}\n")
            f.write(f"Instance Token: {instance_token}\n\n")
            
            # Write history correction details
            if results.get('history_correction', {}).get('history_correction_applied', False):
                f.write("-"*80 + "\n")
                f.write("HISTORY-BASED CORRECTION:\n")
                f.write("-"*80 + "\n\n")
                
                correction = results['history_correction']
                f.write(f"Original distance: {correction.get('original_distance', 0):.2f}m\n")
                f.write(f"Corrected distance: {correction.get('corrected_distance', 0):.2f}m\n")
                f.write(f"Correction method: {correction.get('correction_method', 'N/A')}\n")
                if 'original_cluster' in correction and correction['original_cluster'] is not None:
                    f.write(f"Original cluster: {correction['original_cluster']}\n")
                if 'corrected_cluster' in correction and correction['corrected_cluster'] is not None:
                    f.write(f"Corrected cluster: {correction['corrected_cluster']}\n")
                f.write(f"Reason: {correction.get('reason', 'N/A')}\n")
                
                if 'details' in correction:
                    details = correction['details']
                    f.write(f"\nHistory analysis:\n")
                    f.write(f"  History size: {details.get('history_size', 0)} frames\n")
                    f.write(f"  History median: {details.get('median', 0):.2f}m\n")
                    f.write(f"  History std: {details.get('std', 0):.2f}m\n")
                    f.write(f"  Adaptive threshold: {details.get('adaptive_threshold', 0):.2f}m\n")
                    f.write(f"  Deviation: {details.get('deviation', 0):.2f}m\n")
                    
                    # Write trend analysis if available
                    if 'threshold_info' in details and 'trend_info' in details['threshold_info']:
                        trend_info = details['threshold_info']['trend_info']
                        f.write(f"\n  Trend analysis:\n")
                        f.write(f"    Has history: {trend_info.get('has_history', False)}\n")
                        f.write(f"    Is moving: {trend_info.get('is_moving', False)}\n")
                        f.write(f"    Direction: {trend_info.get('movement_direction', 'unknown')}\n")
                        f.write(f"    Avg change: {trend_info.get('avg_change', 0):.2f}m/frame\n")
                        f.write(f"    Avg abs change: {trend_info.get('avg_abs_change', 0):.2f}m/frame\n")
                        f.write(f"    Trend confidence: {trend_info.get('trend_confidence', 0):.2f}\n")
                f.write("\n")
            
            # Write triple fusion record details
            if results.get('triple_fusion_records'):
                f.write("-"*80 + "\n")
                f.write("TRIPLE FUSION RECORD DETAILS:\n")
                f.write("-"*80 + "\n\n")
                
                for sensor_key, records in results['triple_fusion_records'].items():
                    if records:
                        f.write(f"{sensor_key.upper()}:\n")
                        for record in records:
                            f.write(f"  {record['position'].upper()}:\n")
                            f.write(f"    Sample Token: {record.get('sample_token', 'N/A')}\n")
                            f.write(f"    Timestamp: {record.get('timestamp', 'N/A')}\n")
                            f.write(f"    Distance: {record.get('distance', 'N/A'):.2f}m\n")
                            f.write(f"    Faulty: {record.get('faulty', 'N/A')}\n")
                        f.write("\n")
            
            if results['outlier_info']:
                f.write("-"*80 + "\n")
                f.write("OUTLIER DETECTION:\n")
                f.write("-"*80 + "\n\n")
                
                for sensor, info in results['outlier_info'].items():
                    f.write(f"{sensor.upper()}:\n")
                    f.write(f"  Distance: {info['distance']:.2f}m\n")
                    f.write(f"  Detection methods: {', '.join(info['methods'])}\n")
                    f.write(f"  Original weight: {info['original_weight']:.3f}\n")
                    f.write(f"  Adjusted weight: {info['adjusted_weight']:.3f}\n")
                    
                    if 'iqr' in info['methods']:
                        details = info['details']
                        f.write(f"  IQR Analysis:\n")
                        f.write(f"    - Bounds: [{details['lower_bound']:.2f}, {details['upper_bound']:.2f}]\n")
                        f.write(f"    - Deviation from median: {details['deviation_percent']:.1f}%\n")
                    
                    if 'mad' in info['methods']:
                        details = info['details']
                        mod_z = details.get('modified_z_score')
                        threshold = details.get('threshold')
                        if mod_z is not None:
                            f.write(f"  MAD Analysis:\n")
                            f.write(f"    - Modified Z-score: {mod_z:.2f}\n")
                            f.write(f"    - Threshold: {threshold}\n")
                        else:
                            f.write(f"  MAD Analysis: N/A\n")
                    f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("MATCH INFORMATION:\n")
            f.write("-"*80 + "\n")
            
            sensor_order = [
                ('lidar', 'LIDAR'),
                ('radar_front', 'RADAR_FRONT'),
                ('radar_front_right', 'RADAR_FRONT_RIGHT'),
                ('radar_front_left', 'RADAR_FRONT_LEFT'),
                ('cam_front', 'CAM_FRONT'),
                ('cam_front_right', 'CAM_FRONT_RIGHT'),
                ('cam_front_left', 'CAM_FRONT_LEFT')
            ]
            
            for sensor_key, sensor_name in sensor_order:
                if sensor_key in results['match_info']:
                    match_info = results['match_info'][sensor_key]
                    f.write(f"{sensor_name}:\n")
                    f.write(f"  Status: {match_info.get('status', 'UNKNOWN')}\n")
                    f.write(f"  Details: {match_info.get('details', 'No details')}\n")
                    if 'sample_token' in match_info and match_info['sample_token']:
                        f.write(f"  Sample Token: {match_info['sample_token']}\n")
                    if 'timestamp' in match_info and match_info['timestamp']:
                        f.write(f"  Timestamp: {match_info['timestamp']}\n")
                    if 'num_records_fused' in match_info:
                        f.write(f"  Records Fused: {match_info['num_records_fused']}\n")
                    if 'fusion_method' in match_info:
                        f.write(f"  Fusion Method: {match_info['fusion_method']}\n")
                    f.write("\n")
            
            f.write("-"*80 + "\n")
            f.write("SENSOR READINGS:\n")
            f.write("-"*80 + "\n")
            
            for sensor_key, sensor_name in sensor_order:
                if sensor_key in results['sensors'] and results['sensors'][sensor_key].get('available'):
                    data = results['sensors'][sensor_key]
                    fusion_info = f" ({data.get('fusion_type', 'single')}, {data.get('num_records_fused', 1)} records)" if data.get('fusion_type', '').startswith('triple') else ""
                    outlier_info = f" [OUTLIER, weight × {data.get('outlier_weight_factor', 1.0):.1f}]" if data.get('is_outlier', False) else ""
                    
                    f.write(f"{sensor_name:20} | "
                           f"Distance: {data['distance']:7.2f}m | "
                           f"Weight: {data['weight']:6.3f} | "
                           f"Faulty: {data['faulty']:5}{fusion_info}{outlier_info}\n")
                    
                    # Write fault determination details for triple fusion sensors
                    if data.get('fusion_type', '').startswith('triple') and 'triple_fusion_details' in data:
                        fd = data['triple_fusion_details'].get('fault_determination', {})
                        if fd:
                            f.write(f"{'':20} |   Fault Determination: {fd.get('method', 'unknown')}\n")
                            if 'selected_cluster_faulty_count' in fd:
                                f.write(f"{'':20} |   Selected Cluster: Faulty={fd['selected_cluster_faulty_count']}, Healthy={fd['selected_cluster_healthy_count']}\n")
                else:
                    if sensor_key in results['match_info']:
                        reason = results['match_info'][sensor_key].get('details', 'Unknown')
                        f.write(f"{sensor_name:20} | Not available - {reason}\n")
                    else:
                        f.write(f"{sensor_name:20} | Not available\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("MAJORITY VOTE FUSION RESULTS:\n")
            f.write("-"*80 + "\n")
            
            if results['fused_distance'] is not None:
                f.write(f"Final Fused Distance: {results['fused_distance']:.2f}m\n")
                f.write(f"Number of sensors used: {results['details']['num_sensors']}\n")
                
                # Count outliers
                outlier_count = sum(1 for s, status in results['details'].get('sensor_status', []) 
                                  if status == 'outlier')
                if outlier_count > 0:
                    f.write(f"Sensors with reduced weight (outliers): {outlier_count}\n")
                
                majority_details = results.get('majority_vote_details', {})
                f.write(f"Fusion method: {majority_details.get('method', 'unknown')}\n")
                
                if 'history_used' in majority_details and majority_details['history_used']:
                    f.write(f"History used: Yes ({majority_details.get('history_size', 0)} previous frames)\n")
                
                if majority_details.get('method') == 'majority_vote':
                    f.write(f"\nCluster Analysis:\n")
                    clusters = majority_details.get('cluster_analysis', [])
                    for i, cluster in enumerate(clusters):
                        if i == majority_details.get('selected_cluster'):
                            marker = " [SELECTED]"
                        else:
                            marker = ""
                        f.write(f"  Cluster {i}{marker}:\n")
                        f.write(f"    Sensors: {', '.join(cluster['sensors'])}\n")
                        f.write(f"    Count: {cluster['sensor_count']} (Healthy: {cluster['healthy_count']}, Faulty: {cluster['faulty_count']})\n")
                        f.write(f"    Health Ratio: {cluster['health_ratio']:.2f}\n")
                        f.write(f"    Mean Distance: {cluster['mean_distance']:.2f}m\n")
                        if 'confidence' in cluster:
                            f.write(f"    Confidence: {cluster['confidence']:.2f}\n")
            else:
                f.write("No fused distance available\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"Summary saved to: {txt_path}")
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
    
    else:
        print("Invalid choice. Please enter 1 or 2.")
