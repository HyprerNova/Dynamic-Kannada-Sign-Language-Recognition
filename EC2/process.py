import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Constants
MAX_SEQUENCE_LENGTH = 75
NUM_FEATURES = 195
CLASSES = ['Afternoon', 'Apple', 'April', 'August', 'Banana', 'Day', 'December', 'Evening',
           'Febraury', 'Friday', 'Grapes', 'January', 'July', 'June', 'March', 'May', 'Monday',
           'Morning', 'Night', 'November', 'October', 'Orange', 'Rainy', 'Saturday', 'September',
           'Summer', 'Sunday', 'Thursday', 'Tuesday', 'Valencia_Orange', 'Watermelon', 'Wednesday', 'Winter']
NUM = len(CLASSES)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
from mediapipe.framework.formats import landmark_pb2

def extract_mediapipe_features(video_path, display_video=False):
    sequence_features = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Define the specific pose landmark indices we want to extract
    # (face top, shoulders, arms, wrists, hips)
    # This now includes indices 0-10 (face) AND 11-24 (upper body/hips)
    desired_pose_landmark_indices = set(range(23)) # Indices 0 through 24

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the BGR image to RGB.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False # To improve performance

            # Process the image and find landmarks.
            results = holistic.process(image)

            # Revert to BGR and enable writing for drawing
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # --- Extract only desired features ---
            current_frame_features = []

            # Pose landmarks (only desired upper body points, including the first 11)
            if results.pose_landmarks:
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i in desired_pose_landmark_indices:
                        current_frame_features.extend([landmark.x, landmark.y, landmark.z])

                # Pad if not all desired pose landmarks were detected or if some are skipped
                num_expected_pose_features = len(desired_pose_landmark_indices)
                if len(current_frame_features) < num_expected_pose_features:
                    # Pad with zeros for any missing desired landmarks
                    current_frame_features.extend([0.0] * (num_expected_pose_features - len(current_frame_features)))
            else:
                # If no pose landmarks were detected at all, fill with zeros for all expected pose features
                current_frame_features.extend([0.0] * len(desired_pose_landmark_indices) * 3) # 25 * 4 = 100 zeros

            # Left hand landmarks (21 landmarks, each with x, y, z)
            if results.left_hand_landmarks:
                for landmark in results.left_hand_landmarks.landmark:
                    current_frame_features.extend([landmark.x, landmark.y, landmark.z])
            else:
                current_frame_features.extend([0.0] * 21 * 3) # 63 zeros

            # Right hand landmarks (21 landmarks, each with x, y, z)
            if results.right_hand_landmarks:
                for landmark in results.right_hand_landmarks.landmark:
                    current_frame_features.extend([landmark.x, landmark.y, landmark.z])
            else:
                current_frame_features.extend([0.0] * 21 * 3) # 63 zeros

            # Verify the total number of features matches NUM_FEATURES
            if len(current_frame_features) != NUM_FEATURES:
                print(f"Warning: Feature count mismatch at frame {frame_count}. Expected {NUM_FEATURES}, got {len(current_frame_features)}")
                # Fallback to ensure consistent shape by padding with zeros if mismatch occurs
                if len(current_frame_features) < NUM_FEATURES:
                     current_frame_features.extend([0.0] * (NUM_FEATURES - len(current_frame_features)))
                else: # Truncate if too many (unlikely with this logic but good practice)
                    current_frame_features = current_frame_features[:NUM_FEATURES]


            sequence_features.append(current_frame_features)

            # --- Display Video with specific landmarks ---
            if display_video:
                annotated_image = np.copy(image)

                filtered_pose_display_connections = []
                for conn in mp_holistic.POSE_CONNECTIONS:
                    # Check if BOTH ends of the connection are within your desired_pose_landmark_indices
                    if conn[0].value in desired_pose_landmark_indices and \
                       conn[1].value in desired_pose_landmark_indices:
                        filtered_pose_display_connections.append(conn) # Keep the original enum tuple

                # Draw the filtered pose landmarks
                if results.pose_landmarks:

                    pose_landmarks_to_display = landmark_pb2.NormalizedLandmarkList()
                    for i, landmark in enumerate(results.pose_landmarks.landmark):
                        if i in desired_pose_landmark_indices:
                            pose_landmarks_to_display.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z, visibility=landmark.visibility)
                        else:

                            pose_landmarks_to_display.landmark.add(x=0.0, y=0.0, z=0.0, visibility=0.0)

                    mp_drawing.draw_landmarks(
                        annotated_image,
                        pose_landmarks_to_display, # Use the list with visible desired landmarks and invisible others
                        filtered_pose_display_connections, # Use the connections you filtered
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), # Green points
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # Green lines
                    )

                # Draw hand landmarks (unchanged)
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2), # Blue points
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)  # Blue lines
                    )
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Red points
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)  # Red lines
                    )

                cv2.imshow(f'MediaPipe Holistic - {os.path.basename(video_path)}', annotated_image)
                if cv2.waitKey(5) & 0xFF == 27: # ESC to exit
                    break
            frame_count += 1

    cap.release()
    if display_video:
        cv2.destroyAllWindows()

    if not sequence_features:
        print(f"Warning: No features extracted from {video_path}")
        return np.zeros((0, NUM_FEATURES)) # Return empty array with correct feature dimension

    return np.array(sequence_features)


def standardize_sequence(sequence, max_len, num_features):
    """Pads or truncates a sequence to a fixed length."""
    if len(sequence) > max_len:
        # Truncate the sequence
        return sequence[:max_len]
    elif len(sequence) < max_len:
        # Pad the sequence with zeros
        padding = np.zeros((max_len - len(sequence), num_features))
        return np.vstack((sequence, padding))
    else:
        # Sequence is already the correct length
        return sequence

