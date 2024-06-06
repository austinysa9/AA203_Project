import cv2
import numpy as np

# Load the video
# video_path = 'bulky.MOV'
video_path = 'ball.MOV'
cap = cv2.VideoCapture(video_path)

# Parameters
num_frames_to_stack = 15  # Number of frames to stack together
frame_interval = 5  # Interval between frames to stack

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read the video")
    cap.release()
    exit()

# Get frame dimensions
height, width, _ = frame.shape

# Create an empty image to stack frames
stacked_image = np.zeros((height, width, 3), dtype=np.uint8)

# Initialize frame count
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Only process every nth frame
    if frame_count % frame_interval == 0:
        stacked_image = cv2.addWeighted(stacked_image, 0.5, frame, 0.5, 0)

        # Stop if we have stacked enough frames
        if frame_count // frame_interval >= num_frames_to_stack:
            break

    frame_count += 1

# Release the video capture
cap.release()

# Save the stacked image
output_path_color = 'ball.png'
cv2.imwrite(output_path_color, stacked_image)

output_path_color
