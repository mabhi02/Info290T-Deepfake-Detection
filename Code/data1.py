import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import shutil
from tqdm import tqdm

def create_directory_structure(base_path):
    """Create the directory structure for extracted face frames"""
    for dataset in ['train', 'test']:
        for label in ['fake', 'real']:
            os.makedirs(os.path.join(base_path, dataset, label), exist_ok=True)
    print("Directory structure created successfully.")

def extract_face_frames(source_dir, target_dir, sample_rate=2, max_frames=60):
    """
    Extract face frames from videos at specified sampling rate with a maximum cap
    
    Args:
        source_dir: Path to the source directory containing video files
        target_dir: Path to save the extracted face frames
        sample_rate: Number of frames to extract per second (default: 2)
        max_frames: Maximum number of frames to extract per video (default: 60)
    """
    # Initialize InsightFace face detector
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    
    # Process videos for each dataset (train/test) and label (fake/real)
    for dataset in ['train', 'test']:
        for label in ['fake', 'real']:
            source_path = os.path.join(source_dir, dataset, label)
            target_path = os.path.join(target_dir, dataset, label)
            
            # Skip if source directory doesn't exist
            if not os.path.exists(source_path):
                print(f"Source directory not found: {source_path}")
                continue
            
            # Get list of video files in the source directory
            video_files = [f for f in os.listdir(source_path) if f.endswith('.MOV') or f.endswith('.mov')]
            print(f"Processing {len(video_files)} videos in {dataset}/{label}")
            
            # Process each video
            for video_file in tqdm(video_files, desc=f"{dataset}/{label}"):
                video_path = os.path.join(source_path, video_file)
                video_name = os.path.splitext(video_file)[0]
                
                # Create directory for this video
                video_frame_dir = os.path.join(target_path, video_name)
                os.makedirs(video_frame_dir, exist_ok=True)
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"Could not open video: {video_path}")
                    continue
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Calculate frame sampling interval
                sample_interval = int(fps / sample_rate)
                if sample_interval < 1:
                    sample_interval = 1
                
                # Process the first frame to find the face
                ret, frame = cap.read()
                if not ret:
                    print(f"Could not read first frame from video: {video_path}")
                    cap.release()
                    continue
                
                # Detect faces in the first frame
                faces = face_analyzer.get(frame)
                
                # If no faces detected in the first frame, try a few more frames
                if len(faces) == 0:
                    for _ in range(5):
                        # Skip ahead
                        for _ in range(sample_interval):
                            ret, frame = cap.read()
                            if not ret:
                                break
                        
                        if not ret:
                            break
                        
                        faces = face_analyzer.get(frame)
                        if len(faces) > 0:
                            break
                
                # If still no faces found, skip this video
                if len(faces) == 0:
                    print(f"No faces detected in video: {video_path}")
                    cap.release()
                    continue
                
                # Select the largest face by area
                largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                face_bbox = largest_face.bbox.astype(int)
                
                # Add padding to face bounding box (10% on each side)
                h, w = frame.shape[:2]
                padding_x = int((face_bbox[2] - face_bbox[0]) * 0.1)
                padding_y = int((face_bbox[3] - face_bbox[1]) * 0.1)
                
                x1 = max(0, face_bbox[0] - padding_x)
                y1 = max(0, face_bbox[1] - padding_y)
                x2 = min(w, face_bbox[2] + padding_x)
                y2 = min(h, face_bbox[3] + padding_y)
                
                # Reset video to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                # Extract frames at the specified interval
                frame_index = 0
                saved_frames = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process only frames at the specified interval
                    if frame_index % sample_interval == 0:
                        # Extract face region using the bounding box from first frame
                        try:
                            face_frame = frame[y1:y2, x1:x2]
                            # Save only if we got a valid face region
                            if face_frame.size > 0:
                                output_path = os.path.join(video_frame_dir, f"frame_{saved_frames:03d}.jpg")
                                cv2.imwrite(output_path, face_frame)
                                saved_frames += 1
                                
                                # Stop if we've reached the maximum number of frames
                                if saved_frames >= max_frames:
                                    break
                        except Exception as e:
                            print(f"Error extracting face from frame {frame_index} in {video_path}: {e}")
                    
                    frame_index += 1
                
                # Release video
                cap.release()
                
                # If no frames were saved, remove the empty directory
                if saved_frames == 0:
                    shutil.rmtree(video_frame_dir)
                    print(f"No valid face frames extracted from: {video_path}")

def main():
    # Define paths
    source_dir = 'finalData'
    target_dir = 'faceData'
    
    # Create directory structure
    create_directory_structure(target_dir)
    
    # Extract face frames
    extract_face_frames(source_dir, target_dir, sample_rate=2, max_frames=60)
    
    print("Face extraction completed successfully!")

if __name__ == "__main__":
    main()