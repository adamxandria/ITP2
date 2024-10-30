# Install required packages manually in your terminal before running this code
# pip install facenet-pytorch efficientnet-pytorch filetype face_recognition

import os
import warnings
import gc
import cv2
import filetype
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import face_recognition
import torch
from torchvision.transforms import transforms
from efficientnet_pytorch import EfficientNet as _EfficientNet
import torch.nn as nn

warnings.filterwarnings('ignore') # ignore all warnings
torch.manual_seed(1) # Seed for generating random numbers

# Define paths (adjust these to your local directory structure)
video_path = '/home/deepfake/Documents/Adam/itp2_videoinput/Elon Musks Deep Fake Video Promoting a Crypto Scam.mp4'
output_folder = '/home/deepfake/Documents/Adam/Output/2'

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

DEFAULT_FACE_MIN_CONF = 0.95

class VideoTask:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(select_largest=False, post_process=False, device=self.device)
        self.known_faces = []
        self.tracked_faces = []
        self.next_face_id = 0

    def assign_face_id(self, detected_encoding):
        if not self.tracked_faces:
            face_id = self.next_face_id
            self.tracked_faces.append({'encoding': detected_encoding, 'id': face_id})
            self.next_face_id += 1
            return face_id

        encodings = [face['encoding'] for face in self.tracked_faces]
        distances = face_recognition.face_distance(encodings, detected_encoding)
        min_distance = np.min(distances)
        best_match_index = np.argmin(distances)

        if min_distance < 0.6:
            return self.tracked_faces[best_match_index]['id']
        else:
            face_id = self.next_face_id
            self.tracked_faces.append({'encoding': detected_encoding, 'id': face_id})
            self.next_face_id += 1
            return face_id

    def load_known_faces(self, known_face_images, known_face_names):
        for image_path, name in zip(known_face_images, known_face_names):
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            self.known_faces.append({'encoding': encoding, 'name': name})

    def recognize_faces(self, face_tensor):
        face_image = transforms.ToPILImage()(face_tensor).convert('RGB')
        face_image_np = np.array(face_image)
        detected_face_encoding = face_recognition.face_encodings(face_image_np)

        if not detected_face_encoding:
            return 'Unknown'

        # Compare the detected face to the known faces
        matches = face_recognition.compare_faces(
            [face['encoding'] for face in self.known_faces],
            detected_face_encoding[0]
        )

        face_distances = face_recognition.face_distance(
            [face['encoding'] for face in self.known_faces],
            detected_face_encoding[0]
        )

        # Get the best match (smallest distance)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            recognized_name = self.known_faces[best_match_index]['name']
        else:
            recognized_name = 'Unknown'

        return recognized_name


    def get_boundingbox(self, var_x1, var_y1, var_x2, var_y2):
        var_y2 += (var_y2 - var_y1) / 10
        var_w = var_x2 - var_x1
        var_h = var_y2 - var_y1
        diff_h_w = (var_h - var_w) / 2
        var_x1 -= diff_h_w
        var_x2 += diff_h_w
        return var_x1, var_y1, var_x2, var_y2

    def crop_face(self, frame, prob, boxes):
        crop_frame = []
        if prob >= DEFAULT_FACE_MIN_CONF:
            xmin, ymin, xmax, ymax = boxes
            xmin, ymin, xmax, ymax = self.get_boundingbox(xmin, ymin, xmax, ymax)
            crop_frame = frame.crop((xmin, ymin, xmax, ymax))  # Crop and Resize
            crop_frame = crop_frame.resize((224, 224))
        return crop_frame

    
    def image_face(self, frame, face_min_conf):
        boxes, prob = self.mtcnn.detect(frame)
        crop_faces = []
        face_boxes = []

        if prob is not None:
          for i, (p, box) in enumerate(zip(prob, boxes)):
            if p >= face_min_conf:
              #compare with the previous detected faces, if more than one face is detected, choose the one with the smallest confidence score (largest probability)
              best_box = box
              best_conf = p
              best_cropped_face = self.crop_face(frame, best_conf, best_box)

              for j, (p2, box2) in enumerate(zip(prob[i+1:], boxes[i+1:])):
                #check if the boxes are similar (faces close together)
                if p2 >= face_min_conf:
                  prelim_cropped_face = self.crop_face(frame, p2, box2)

                  if np.allclose(best_cropped_face, prelim_cropped_face, atol=20):
                  #choose the face with the highest confidence (smaller probability value)
                      if p2 < best_conf:
                          best_box = box2
                          best_conf = p2
                          best_cropped_face = prelim_cropped_face

              #once we have the best match, proceed to crop the size
              #cropped_face = self.crop_face(frame, best_conf, best_box)
              cropped_face = best_cropped_face
              crop_faces.append(cropped_face)
              face_boxes.append(np.array(best_box))  # Store the best bounding box as an array (float)

        return crop_faces, face_boxes

    
    def image_norm_tensor(self, frame, face_min_conf):
        crop_faces, face_boxes = self.image_face(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), face_min_conf)
        tensor_faces = [transforms.ToTensor()(face) for face in crop_faces]  # Convert cropped faces to tensors
        return tensor_faces, face_boxes


    
    def video_process(self, infile, face_min_conf, max_fr):
        length = 0
        face_data_list = []  # List to store all face data

        if filetype.is_video(infile):
            cap = cv2.VideoCapture(infile)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frame = int(max_fr * length)

            # Correct frame list generation
            # the_list = np.round(np.linspace(0, length, max_frame, endpoint=False)).astype(int)
            the_list = list(range(0, length))  # Process every frame in sequence
            counter_frame = 0  # Frame counter
            frame_number = 0  # Frame number for saving

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    if counter_frame in the_list:
                      ## Batch
                        crop_faces, face_boxes = self.image_norm_tensor(frame, face_min_conf)

                        # Store faces, frame number, and bounding boxes in the list
                        for i, (face, box) in enumerate(zip(crop_faces, face_boxes)):

                            # recognized_name = self.recognize_faces(face)

                            face_image = transforms.ToPILImage()(face).convert('RGB')
                            face_image_np = np.array(face_image)

                            # Encode the detected face
                            detected_face_encoding = face_recognition.face_encodings(face_image_np)
                            if detected_face_encoding:
                                face_id= self.assign_face_id(detected_face_encoding[0])
                                print(f"Assigned ID {face_id} to detected face")



                            # Add to the face data list
                            face_data_list.append({
                                'frame_number': counter_frame,  # Label each face with frame number
                                'face_tensor': face,  # Tensor of the cropped face
                                'bounding_box': box,  # Bounding box coordinates of the face
                                'face_id': face_id
                            })

                            # Convert tensor to PIL image (removing unusual color maps)
                            pil_image = transforms.ToPILImage()(face)

                            # Save the frame as an image
                            frame_filename = f'{output_folder}/frame_{frame_number:04d}_face_{i}_{face_id}.jpg'
                            pil_image.save(frame_filename)
                            print(f'Saved {frame_filename} - assigned ID: {face_id}')
                        frame_number += 1
                except Exception as e:
                    print(f"Error processing frame {counter_frame}: {e}")
                counter_frame += 1
            cap.release()  # Release the video capture object

        return face_data_list  # Return the list containing all face data

# Initialize and process video
video_task = VideoTask()
frame_percent = 0.1
face_data_list = video_task.video_process(video_path, DEFAULT_FACE_MIN_CONF, frame_percent)

# Output for verification
for face_data in face_data_list:
    print(f"Frame: {face_data['frame_number']}, Bounding Box: {face_data['bounding_box']}")
