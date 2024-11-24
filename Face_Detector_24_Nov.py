import os
import warnings
import cv2
import filetype
import numpy as np
from facenet_pytorch import MTCNN
import face_recognition
import torch
from torchvision.transforms import transforms
from PIL import Image
import concurrent.futures


DEFAULT_FACE_MIN_CONF = 0.95

warnings.filterwarnings("ignore")  
torch.manual_seed(1)  

video_path = '/home/deepfake/Documents/Adam/dataset/multiple-entity/4-mins-video.mp4'
output_folder = '/home/deepfake/Documents/Adam/Output/10'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


class FaceDetection:
    def __init__(self):
        """
        Initialize the FaceDetection class.
        - Set the device to CUDA if available, otherwise CPU.
        - Load the MTCNN model for face detection.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(select_largest=True, post_process=False, device=self.device)

    def get_boundingbox(self, var_x1, var_y1, var_x2, var_y2):
        """
        Adjust the bounding box for the detected face to ensure consistent cropping.
        Args:
            var_x1, var_y1: Top-left corner of the bounding box.
            var_x2, var_y2: Bottom-right corner of the bounding box.
        Returns:
            Adjusted bounding box coordinates as integers.
        """
        var_y2 += (var_y2 - var_y1) / 10
        var_w = var_x2 - var_x1
        var_h = var_y2 - var_y1
        diff_h_w = (var_h - var_w) / 2
        var_x1 -= diff_h_w
        var_x2 += diff_h_w
        return int(var_x1), int(var_y1), int(var_x2), int(var_y2)

    def crop_face(self, frame, box):
        """
        Crop and resize the face based on the detected bounding box.
        Args:
            frame: The video frame as a numpy array.
            box: Bounding box for the face [x1, y1, x2, y2].
        Returns:
            Cropped and resized face image as a numpy array.
        """
        xmin, ymin, xmax, ymax = self.get_boundingbox(*box)
        cropped_face = frame[ymin:ymax, xmin:xmax]  
        cropped_face = cv2.resize(cropped_face, (224, 224))  
        return cropped_face

    def image_face(self, frame, face_min_conf):
        """
        Detect faces in a single video frame.
        Args:
            frame: A single video frame.
            face_min_conf: Minimum confidence threshold for detection.
        Returns:
            crop_faces: List of cropped face images.
            face_boxes: List of bounding boxes for detected faces.
        """
        boxes, probs = self.mtcnn.detect(frame)
        crop_faces = []
        face_boxes = []

        if probs is not None:
            for p, box in zip(probs, boxes):
                if p >= face_min_conf:
                    cropped_face = self.crop_face(frame, box)
                    crop_faces.append(cropped_face)
                    face_boxes.append(box)

        return crop_faces, face_boxes

    def video_process(self, infile, face_min_conf, frame_rate_reduction=0.1):
        """
        Process the input video and detect faces in every nth frame.
        Args:
            infile: Path to the input video file.
            face_min_conf: Minimum confidence for face detection.
            frame_rate_reduction: Fraction of frames to process (e.g., 0.1 for every 10th frame).
        Returns:
            face_data_list: List of dictionaries containing cropped face images and bounding boxes.
        """
        face_data_list = []  

        if filetype.is_video(infile):
            cap = cv2.VideoCapture(infile)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = int(1 / frame_rate_reduction) 
            counter_frame = 0  

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if counter_frame % frame_interval == 0:
                    try:
                        crop_faces, face_boxes = self.image_face(frame, face_min_conf)
                        frame_faces = {}

                        for i, (face, box) in enumerate(zip(crop_faces, face_boxes)):
                            frame_faces[i] = {
                                'cropped_img': transforms.ToTensor()(face),
                                'face_box': [int(coord) for coord in box]
                            }

                            
                            """
                            Save the cropped face to the specified output folder.
                            Args:
                                face: The cropped face image (numpy array).
                                save_path: The directory where the face image will be saved.
                                counter_frame: The frame number the face was detected in.
                                i: The index of the face in the current frame.
                            """

                            # Print face details for verification
                            
                            # file_name = f"frame_{counter_frame}_face_{i}.jpg"
                            # save_path = os.path.join(output_folder, file_name)
                            # cv2.imwrite(save_path, face)  
                            # print(f"Saved cropped face to: {save_path}")

                        if frame_faces:
                            face_data_list.append(frame_faces)
                            # print(f"Frame {counter_frame} Processed: {frame_faces}")
                    except Exception as e:
                        print(f"Error processing frame {counter_frame}: {e}")
                counter_frame += 1

            cap.release()  

        return face_data_list


class FaceRecognition:
    def __init__(self):
        """
        Initialize the FaceRecognition class.
        - Store known face encodings and IDs.
        - Initialize ID counter for new faces.
        """
        self.known_faces = []  
        self.next_face_id = 0  

    def assign_face_id(self, detected_encoding):
        """
        Assign a unique ID to a detected face based on its encoding.
        Args:
            detected_encoding: The face encoding vector.
        Returns:
            face_id: Unique ID for the detected face.
        """
        if not self.known_faces:
            face_id = self.next_face_id
            self.known_faces.append({'encoding': detected_encoding, 'id': face_id})
            self.next_face_id += 1
            return face_id

        encodings = [face['encoding'] for face in self.known_faces]
        distances = face_recognition.face_distance(encodings, detected_encoding)
        min_distance = np.min(distances)
        best_match_index = np.argmin(distances)

        if min_distance < 0.6:
            return self.known_faces[best_match_index]['id']
        else:
            face_id = self.next_face_id
            self.known_faces.append({'encoding': detected_encoding, 'id': face_id})
            self.next_face_id += 1
            return face_id

    def recognise_faces(self, detected_faces):
        """
        Recognize detected faces and assign IDs.
        Args:
            detected_faces: List of detected face data including cropped images and bounding boxes.
        Returns:
            identity_list: List of recognized faces with assigned IDs and metadata.
        """
        cropped_faces = []
        identity_list = []
        existing_uuid = []

        for frame_faces in detected_faces:
            for i, face_data in frame_faces.items():
                face_tensor = face_data['cropped_img']
                face_bgr = face_tensor.mul(255).byte().permute(1, 2, 0).numpy().astype(np.uint8)
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                xmin, ymin, xmax, ymax = face_data['face_box']
                face_location = (ymin, xmax, ymax, xmin)
                face_locations = [face_location]
                if face_locations:
                    detected_face_encoding = face_recognition.face_encodings(face_rgb)
                else:
                    detected_face_encoding = []

                if detected_face_encoding:
                    face_id = self.assign_face_id(detected_face_encoding[0])

                    # Save the recognized face for verification
                    # file_name = f"uuid_{face_id}.jpg"
                    # save_path = os.path.join(output_folder, file_name)
                    # cv2.imwrite(save_path, face_bgr)

                    cropped_face = {
                        'cropped_img': face_tensor,
                        'uuid': face_id,
                        'face_box': face_data['face_box']
                    }
                    cropped_faces.append(cropped_face)

                    # Save first instance of identity to identity_list and add uuid to existing_uuid
                    if face_id not in existing_uuid:
                        identity_data = {
                            face_id: face_tensor
                        }
                        identity_list.append(identity_data)
                        existing_uuid.append(face_id)

        return cropped_faces, identity_list


# Initialise and process video for detection and recognition
face_detection = FaceDetection()
face_recognition_task = FaceRecognition()
frame_percent = 0.1

with concurrent.futures.ThreadPoolExecutor() as executor:
    detection_future = executor.submit(face_detection.video_process, video_path, DEFAULT_FACE_MIN_CONF, frame_percent)

    detected_faces = detection_future.result()
    recognition_future = executor.submit(face_recognition_task.recognise_faces, detected_faces)
    cropped_faces, identity_list = recognition_future.result()

#Output for verification
print("\nIdentity List Output:")
for identity in identity_list:
    print(identity)


