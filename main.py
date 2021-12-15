import mediapipe as mp
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sub_system_2.gestureClassification import GestureClassifierMLP,GestureClassifierKNN
import tensorflow.keras.backend as K
import sys
from joblib import load as jlLoad
import os


if __name__=="__main__":
    drawing = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles
    hands = mp.solutions.hands
    mp_joints = [hands.HandLandmark.WRIST,hands.HandLandmark.THUMB_MCP,hands.HandLandmark.THUMB_IP,hands.HandLandmark.THUMB_TIP,hands.HandLandmark.INDEX_FINGER_MCP,hands.HandLandmark.INDEX_FINGER_PIP,hands.HandLandmark.INDEX_FINGER_DIP,hands.HandLandmark.INDEX_FINGER_TIP,hands.HandLandmark.MIDDLE_FINGER_MCP,hands.HandLandmark.MIDDLE_FINGER_PIP,hands.HandLandmark.MIDDLE_FINGER_DIP,hands.HandLandmark.MIDDLE_FINGER_TIP,hands.HandLandmark.RING_FINGER_MCP,hands.HandLandmark.RING_FINGER_PIP,hands.HandLandmark.RING_FINGER_DIP,hands.HandLandmark.RING_FINGER_TIP,hands.HandLandmark.PINKY_MCP,hands.HandLandmark.PINKY_PIP,hands.HandLandmark.PINKY_DIP,hands.HandLandmark.PINKY_TIP]
    
    #gesture_clf = GestureClassifier('sub_system_2/models/mlp_reg','sub_system_2/encoder_reg.obj')
    gesture_clf = GestureClassifierKNN(os.path.join('sub_system_2','models','knn.obj'))


    if len(sys.argv) == 2:
        id = sys.argv[1]
    else:
        id = 0
    cap = cv2.VideoCapture(id) 
    with hands.Hands(
                model_complexity=0,
                static_image_mode=(id!=0),
                max_num_hands=1,
                min_detection_confidence=0.5
        ) as hands_detector:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    if id == 0:
                        continue
                    else:
                        break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands_detector.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                image_height, image_width, _ = image.shape
                image_annot = image.copy()
                joints_corrdinate = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        for mp_joint in mp_joints:
                            joints_corrdinate.append((hand_landmarks.landmark[mp_joint].x * image_width,hand_landmarks.landmark[mp_joint].y * image_height))
                        for c in joints_corrdinate:
                            cv2.circle(image_annot,(int(c[0]),int(c[1])),5,(255,0,0))
                gesture = gesture_clf.predict(np.array(joints_corrdinate))
                print(gesture)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('Model predictions',cv2.flip(image_annot,1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()