Face Detection and Recognition System
This project implements a face detection and recognition system using MTCNN for face detection and face_recognition for encoding and recognizing faces. The system processes videos to detect faces, crops them, and assigns unique IDs to recognized faces.

Features
Detects faces in video frames using MTCNN.
Crops and resizes detected faces for further processing.
Assigns unique IDs (uuid) to recognized faces based on facial encoding.
Supports parallel processing using multithreading for efficient video processing.
Prerequisites
Python 3.7 or later
Libraries:
torch
numpy
opencv-python
facenet-pytorch
face_recognition
Pillow
filetype
Install the required dependencies using the following command:

bash
Copy code
pip install torch numpy opencv-python facenet-pytorch face_recognition Pillow filetype
Usage
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Place your input video file in the desired directory and update the video_path variable in the script with the video file path:

python
Copy code
video_path = '/path/to/your/video.mp4'
Specify the output folder where cropped face images and recognized face images will be saved:

python
Copy code
output_folder = '/path/to/output/folder'
Run the script:

bash
Copy code
python face_detection_recognition.py
The output will include:

Cropped face images saved to the specified output_folder.
Unique IDs assigned to recognized faces printed in the terminal.
Bounding box information for each detected face in the video.
Sample Output
The script prints the recognized face data in the terminal:

plaintext
Copy code
Identity List Output:
{'cropped_img': tensor(...), 'uuid': 0, 'face_box': [101, 10, 211, 41]}
{'cropped_img': tensor(...), 'uuid': 1, 'face_box': [120, 50, 230, 80]}
...
Each face is assigned a unique uuid based on its encoding, and the cropped face image tensor and bounding box are displayed.

Customization
Adjust Detection Confidence: Modify the DEFAULT_FACE_MIN_CONF variable to adjust the confidence threshold for face detection:

python
Copy code
DEFAULT_FACE_MIN_CONF = 0.95
Frame Processing Rate: Adjust the frame_rate_reduction parameter in the video_process method to process more or fewer frames:

python
Copy code
frame_rate_reduction = 0.1  # Processes every 10th frame
Save Recognized Faces: Uncomment the code in the recognise_faces method to save recognized faces to the output_folder.

Acknowledgments
MTCNN for face detection.
face_recognition for face encoding and recognition.
License
This project is licensed under the MIT License. See the LICENSE file for details.
