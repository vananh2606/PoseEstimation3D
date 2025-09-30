import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean
from dtaidistance import dtw

class PoseVisualizer:
    def __init__(self):
        # Human3.6M skeleton connections (parent -> child)
        self.skeleton = [
            [0, 1], [1, 2], [2, 3],  # right leg
            [0, 4], [4, 5], [5, 6],  # left leg  
            [0, 7], [7, 8], [8, 9], [9, 10],  # spine to head
            [8, 11], [11, 12], [12, 13],  # left arm
            [8, 14], [14, 15], [15, 16]   # right arm
        ]
        
        self.joint_names = [
            'hip', 'right_hip', 'right_knee', 'right_foot',
            'left_hip', 'left_knee', 'left_foot', 
            'spine', 'thorax', 'neck', 'head',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]

    def transform_pose(self, pose_3d):
        """Đảo trục XYZ để hiển thị chuẩn hơn"""
        x = pose_3d[:, 0]
        y = pose_3d[:, 1]
        z = pose_3d[:, 2]
        # Chuyển hệ trục
        return np.stack([x, z, -y], axis=1)
    
    def plot_pose_3d(self, pose_3d, title="3D Pose"):
        """Plot single 3D pose"""
        pose_3d = self.transform_pose(pose_3d)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot joints
        ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], 
                  c='red', s=50, alpha=0.8)
        
        # Plot skeleton connections
        for connection in self.skeleton:
            start_joint, end_joint = connection
            ax.plot([pose_3d[start_joint, 0], pose_3d[end_joint, 0]],
                   [pose_3d[start_joint, 1], pose_3d[end_joint, 1]], 
                   [pose_3d[start_joint, 2], pose_3d[end_joint, 2]], 
                   'b-', alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([pose_3d[:, 0].max() - pose_3d[:, 0].min(),
                             pose_3d[:, 1].max() - pose_3d[:, 1].min(),
                             pose_3d[:, 2].max() - pose_3d[:, 2].min()]).max() / 2.0
        
        mid_x = (pose_3d[:, 0].max() + pose_3d[:, 0].min()) * 0.5
        mid_y = (pose_3d[:, 1].max() + pose_3d[:, 1].min()) * 0.5
        mid_z = (pose_3d[:, 2].max() + pose_3d[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.show()
    
    def create_3d_animation(self, poses_3d, output_path="pose_animation.gif", fps=30):
        """Create 3D animation from pose sequence"""
        poses_3d = [self.transform_pose(pose) for pose in poses_3d]
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame_idx):
            ax.clear()
            pose = poses_3d[frame_idx]
            
            # Plot joints
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], 
                      c='red', s=50, alpha=0.8)
            
            # Plot skeleton
            for connection in self.skeleton:
                start_joint, end_joint = connection
                ax.plot([pose[start_joint, 0], pose[end_joint, 0]],
                       [pose[start_joint, 1], pose[end_joint, 1]], 
                       [pose[start_joint, 2], pose[end_joint, 2]], 
                       'b-', alpha=0.7, linewidth=2)
            
            # Set consistent axis limits
            all_poses = np.concatenate(poses_3d, axis=0)
            max_range = np.array([all_poses[:, 0].max() - all_poses[:, 0].min(),
                                 all_poses[:, 1].max() - all_poses[:, 1].min(),
                                 all_poses[:, 2].max() - all_poses[:, 2].min()]).max() / 2.0
            
            mid_x = (all_poses[:, 0].max() + all_poses[:, 0].min()) * 0.5
            mid_y = (all_poses[:, 1].max() + all_poses[:, 1].min()) * 0.5
            mid_z = (all_poses[:, 2].max() + all_poses[:, 2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'3D Pose Animation - Frame {frame_idx + 1}/{len(poses_3d)}')
        
        # Create animation
        ani = animation.FuncAnimation(fig, animate, frames=len(poses_3d), 
                                    interval=1000/fps, repeat=True)
        
        # Save as GIF
        ani.save(output_path, writer='pillow', fps=fps)
        print(f"Animation saved to: {output_path}")
        
        return ani

class AdvancedMotionAnalyzer:
    def __init__(self):
        self.joint_names = [
            'hip', 'right_hip', 'right_knee', 'right_foot',
            'left_hip', 'left_knee', 'left_foot',
            'spine', 'thorax', 'neck', 'head', 
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]
    
    def detect_repetitive_motions(self, poses_3d, joint_idx=2, threshold=0.8):
        """Detect repetitive motions using autocorrelation"""
        # Extract trajectory of specific joint
        trajectory = poses_3d[:, joint_idx, :]
        
        # Calculate distance from starting position
        distances = [euclidean(trajectory[i], trajectory[0]) 
                    for i in range(len(trajectory))]
        
        # Find peaks (local maxima) in distance signal
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(distances, height=np.std(distances))
        
        # Estimate period
        if len(peaks) > 1:
            periods = np.diff(peaks)
            avg_period = np.mean(periods) if len(periods) > 0 else 0
            repetitions = len(peaks)
        else:
            avg_period = 0
            repetitions = 0
        
        return {
            'repetitions': repetitions,
            'avg_period': avg_period,
            'distances': distances,
            'peaks': peaks.tolist()
        }
    
    def calculate_motion_smoothness(self, poses_3d):
        """Calculate motion smoothness using jerk (3rd derivative)"""
        # Calculate velocities
        velocities = np.diff(poses_3d, axis=0)
        
        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0)
        
        # Calculate jerk (3rd derivative)
        jerk = np.diff(accelerations, axis=0)
        
        # Smoothness metric (lower jerk = smoother motion)
        smoothness = -np.mean(np.linalg.norm(jerk, axis=2))
        
        return {
            'smoothness_score': smoothness,
            'avg_velocity': np.mean(np.linalg.norm(velocities, axis=2)),
            'avg_acceleration': np.mean(np.linalg.norm(accelerations, axis=2)),
            'avg_jerk': np.mean(np.linalg.norm(jerk, axis=2))
        }
    
    def compare_motion_patterns(self, poses_3d_1, poses_3d_2, joint_idx=2):
        """Compare two motion patterns using DTW"""
        # Extract trajectories
        traj_1 = poses_3d_1[:, joint_idx, :]
        traj_2 = poses_3d_2[:, joint_idx, :]
        
        # Calculate DTW distance
        distance = dtw.distance(traj_1.flatten(), traj_2.flatten())
        
        # Normalize by sequence lengths
        normalized_distance = distance / (len(traj_1) + len(traj_2))
        
        return {
            'dtw_distance': distance,
            'normalized_distance': normalized_distance,
            'similarity_score': 1.0 / (1.0 + normalized_distance)
        }
    
    def analyze_balance_stability(self, poses_3d):
        """Analyze balance and stability"""
        # Center of mass approximation (average of all joints)
        center_of_mass = np.mean(poses_3d, axis=1)
        
        # Calculate center of mass trajectory
        com_trajectory = center_of_mass
        
        # Stability metrics
        com_displacement = np.std(com_trajectory, axis=0)
        total_displacement = np.linalg.norm(com_displacement)
        
        # Sway area (area of ellipse containing 95% of COM positions)
        from scipy.stats import chi2
        confidence_level = 0.95
        s = -2 * np.log(1 - confidence_level)
        
        cov_matrix = np.cov(com_trajectory[:, 0], com_trajectory[:, 2])  # X-Z plane
        eigenvals, _ = np.linalg.eig(cov_matrix)
        sway_area = np.pi * np.sqrt(s * eigenvals[0] * eigenvals[1])
        
        return {
            'com_displacement': total_displacement,
            'sway_area': sway_area,
            'stability_score': 1.0 / (1.0 + total_displacement)
        }
    
    def comprehensive_analysis(self, poses_3d, fps=30):
        """Run comprehensive motion analysis"""
        results = {}
        
        # Basic motion metrics
        motion_smoothness = self.calculate_motion_smoothness(poses_3d)
        results['motion_smoothness'] = motion_smoothness
        
        # Repetitive motion detection
        repetitive_motion = self.detect_repetitive_motions(poses_3d)
        results['repetitive_motion'] = repetitive_motion
        
        # Balance analysis
        balance_analysis = self.analyze_balance_stability(poses_3d)
        results['balance_stability'] = balance_analysis
        
        # Motion statistics
        results['duration'] = len(poses_3d) / fps
        results['total_frames'] = len(poses_3d)
        
        return results

# Load từ JSON thay vì NPY
with open("res/output/output_3d_poses.json", "r") as f:
    data = json.load(f)

# poses_3d shape = (people, frames, joints, 3)
poses_3d = np.array(data["poses_3d"])

# Chọn person 0 (hoặc 1 nếu bạn muốn)
person_idx = 1
person_poses = poses_3d[person_idx]   # shape = (frames, joints, 3)

# --- Visualization ---
visualizer = PoseVisualizer()

# Vẽ frame đầu tiên
visualizer.plot_pose_3d(person_poses[0], title="First Pose (Person 0)")

# Tạo animation (sample mỗi 5 frame cho nhanh)
sampled_poses = person_poses[::5]
visualizer.create_3d_animation(sampled_poses, "motion_analysis.gif", fps=10)

# --- Advanced Analysis ---
analyzer = AdvancedMotionAnalyzer()
analysis_results = analyzer.comprehensive_analysis(person_poses, fps=30)

print("\n=== COMPREHENSIVE MOTION ANALYSIS ===")
print(f"Duration: {analysis_results['duration']:.2f} seconds")
print(f"Motion smoothness score: {analysis_results['motion_smoothness']['smoothness_score']:.4f}")
print(f"Repetitions detected: {analysis_results['repetitive_motion']['repetitions']}")
print(f"Balance stability score: {analysis_results['balance_stability']['stability_score']:.4f}")
print(f"Sway area: {analysis_results['balance_stability']['sway_area']:.4f}")
