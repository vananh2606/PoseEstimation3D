import cv2
import numpy as np
import mediapipe as mp

class SideBySidePoseComparison:
    def __init__(self, reference_video_path):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load reference video
        self.reference_video_path = reference_video_path
        self.ref_cap = cv2.VideoCapture(reference_video_path)
        self.ref_fps = self.ref_cap.get(cv2.CAP_PROP_FPS)
        
        # Get video info
        total_frames = int(self.ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Reference video: {total_frames} frames, {self.ref_fps} FPS")
        
    def _extract_keypoints(self, image):
        """Extract pose keypoints from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        keypoints = None
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y])
            keypoints = np.array(keypoints)
        
        return keypoints, results
    
    def _calculate_similarity(self, pose1, pose2):
        """Calculate similarity between two poses"""
        if pose1 is None or pose2 is None:
            return 0.0
        
        distance = np.linalg.norm(pose1 - pose2)
        similarity = max(0, 1 - distance)
        return similarity
    
    def _draw_pose(self, frame, results, color=(0, 255, 0)):
        """Draw pose landmarks on frame"""
        if results.pose_landmarks:
            # Draw landmarks with custom color
            landmarks = results.pose_landmarks.landmark
            h, w, _ = frame.shape
            
            # Draw connections
            connections = self.mp_pose.POSE_CONNECTIONS
            for connection in connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                
                if start.visibility > 0.5 and end.visibility > 0.5:
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_point, end_point, color, 2)
            
            # Draw keypoints
            for landmark in landmarks:
                if landmark.visibility > 0.5:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame
    
    def _create_display(self, ref_frame, ref_results, user_frame, user_results, similarity):
        """Create side-by-side display"""
        # Resize frames
        height = 480
        width = 640
        
        if ref_frame is not None:
            ref_display = cv2.resize(ref_frame, (width, height))
            ref_display = self._draw_pose(ref_display, ref_results, color=(0, 0, 255))  # Red
        else:
            ref_display = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(ref_display, "No Reference", (width//2-100, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        user_display = cv2.resize(user_frame, (width, height))
        user_display = self._draw_pose(user_display, user_results, color=(0, 255, 0))  # Green
        
        # Create combined display
        combined = np.hstack([ref_display, user_display])
        
        # Add labels
        cv2.putText(combined, "Reference", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(combined, "Your Pose", (width + 20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add similarity score
        score_text = f"Similarity: {int(similarity * 100)}%"
        score_color = (0, 255, 0) if similarity > 0.7 else (0, 165, 255) if similarity > 0.4 else (0, 0, 255)
        
        # Score background
        (text_width, text_height), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.rectangle(combined, (width - 50, height - 80), 
                     (width + text_width + 50, height - 20), (0, 0, 0), -1)
        
        # Score text
        cv2.putText(combined, score_text, (width - 30, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, score_color, 3)
        
        return combined
    
    def run(self, camera_index=0):
        """Run side-by-side comparison"""
        user_cap = cv2.VideoCapture(camera_index)
        user_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        user_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting side-by-side pose comparison...")
        print("Controls: 'q' to quit, 'r' to restart reference video, SPACE to pause")
        
        paused = False
        
        try:
            while True:
                if not paused:
                    # Read user frame
                    ret_user, user_frame = user_cap.read()
                    if not ret_user:
                        break
                    
                    # Flip for mirror effect
                    user_frame = cv2.flip(user_frame, 1)
                    
                    # Read reference frame
                    ret_ref, ref_frame = self.ref_cap.read()
                    if not ret_ref:
                        # Restart reference video
                        self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_ref, ref_frame = self.ref_cap.read()
                    
                    # Extract poses
                    user_keypoints, user_results = self._extract_keypoints(user_frame)
                    
                    ref_keypoints, ref_results = None, None
                    if ret_ref:
                        ref_keypoints, ref_results = self._extract_keypoints(ref_frame)
                    
                    # Calculate similarity
                    similarity = self._calculate_similarity(user_keypoints, ref_keypoints)
                
                # Create display
                display = self._create_display(ref_frame if ret_ref else None, ref_results, 
                                             user_frame, user_results, similarity)
                
                # Show display
                cv2.imshow('Pose Comparison - Reference vs Your Pose', display)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Restart reference video
                    self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print("Reference video restarted")
                elif key == ord(' '):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            user_cap.release()
            self.ref_cap.release()
            cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    # Replace with your reference video path
    reference_video = "res/input/DAN-DO.mp4"
    
    comparison = SideBySidePoseComparison(reference_video)
    comparison.run(camera_index=0)