ITP2
Overview

The ITP2 project is designed for media analysis, with a focus on detecting and recognizing faces in video files. This project leverages face detection and recognition techniques to analyze video frames, assign unique face IDs, and track faces throughout the video. The core functionalities include detecting faces, generating tensors for face data, assigning unique IDs to detected faces, and saving each processed frame with detailed face information.
Requirements

Please refer to the requirements.txt file for all necessary dependencies. Install them using the following command:

pip install -r requirements.txt

Features

    Face Detection and Recognition: Detects all faces in each video frame and assigns unique IDs to each face.
    Face Tracking: Maintains consistency in face identification across frames.
    Tensor Conversion: Converts detected faces into tensors for further processing and analysis.
    Frame Processing: Processes frames sequentially or in batches, based on the specified percentage, and saves annotated frames.

Installation

    Clone this repository to your local machine:

git clone <repository_url>

Navigate to the project directory:

cd ITP2

Install the required packages:

    pip install -r requirements.txt

Usage

To start using the VideoTask class for face detection and tracking, follow these steps:

    Load Known Faces (Optional): You can load known face images if you want the system to recognize specific faces in the video.

# Initialize the VideoTask class
video_task = VideoTask()

# Load known faces (update paths and names accordingly)
known_faces_paths = ['path/to/face1.jpg', 'path/to/face2.jpg']
known_face_names = ['Person 1', 'Person 2']
video_task.load_known_faces(known_faces_paths, known_face_names)

Process Video: Specify the input video path, confidence threshold for face detection, and the percentage of frames to process.

# Set parameters
video_path = 'path/to/video.mp4'
DEFAULT_FACE_MIN_CONF = 0.9  # Confidence threshold
frame_percent = 0.1  # Percentage of video frames to process

# Process the video
face_data_list = video_task.video_process(video_path, DEFAULT_FACE_MIN_CONF, frame_percent)

Display Results: After processing, you can view the details of detected faces:

    for face_data in face_data_list:
        print(f"Frame: {face_data['frame_number']}, Face ID: {face_data['face_id']}, Bounding Box: {face_data['bounding_box']}, Face Tensor: {face_data['face_tensor']}")

Output

Processed frames with detected faces are saved in the specified output directory with unique names based on the frame number and face ID.


