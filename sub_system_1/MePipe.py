import os

import cv2
import mediapipe as mp
import pandas
import pandas as pd

import func


def setting_mediapipe():
    """

    Run this function to set up mediapipe library.

    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    return mp_drawing, mp_drawing_styles, mp_hands


def detect_hand_video():
    """
    Using MediaPipe to capture video input.
    Could output landmarks in real-time.
    Returns: No return.

    """
    drawing, styles, hands = setting_mediapipe()
    cap = cv2.VideoCapture(0)
    with hands.Hands(
            model_complexity=0,
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
    ) as hands_detector:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        hands.HAND_CONNECTIONS,
                        styles.get_default_hand_landmarks_style(),
                        styles.get_default_hand_connections_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


def detect_hand(video, gesture, video_idx):
    """
    From given video, detect hand landmarks and give back a dataframe containing the corrdinates.
    Args:
        video: Input video filename.
        gesture: Corresponding gestures.
        video_idx: Index of video, used for output dataframe.

    Returns: A dataframe containing corrdinates.

    """
    drawing, styles, hands = setting_mediapipe()
    with hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
    ) as hands_detector:
        frames = func.read_record(video)
        name = ['frame', 'gesture', 'joint', 'video_idx', 'x', 'y']
        joints = ['root', 'thumb_1', 'thumb_2', 'thumb_3', 'index_1', 'index_2', 'index_3',
                  'index_4', 'middle_1', 'middle_2', 'middle_3', 'middle_4', 'ring_1',
                  'ring_2', 'ring_3', 'ring_4', 'pinky_1', 'pinky_2', 'pinky_3', 'pinky_4']
        joints_corrdinate = []
        data = []
        for idx, file in enumerate(frames):
            image = cv2.flip(file, 1)
            results = hands_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
                continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.WRIST].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.WRIST].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.THUMB_MCP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.THUMB_MCP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.THUMB_IP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.THUMB_IP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.THUMB_TIP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.THUMB_TIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.INDEX_FINGER_MCP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.INDEX_FINGER_MCP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.INDEX_FINGER_PIP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.INDEX_FINGER_PIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.INDEX_FINGER_DIP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.INDEX_FINGER_DIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.INDEX_FINGER_TIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width,
                                          hand_landmarks.landmark[
                                              hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_MCP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_MCP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_PIP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_PIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_DIP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_DIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_TIP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.RING_FINGER_TIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.PINKY_MCP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.PINKY_MCP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.PINKY_PIP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.PINKY_PIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.PINKY_DIP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.PINKY_DIP].y * image_height))
                joints_corrdinate.append((hand_landmarks.landmark[hands.HandLandmark.PINKY_TIP].x * image_width,
                                          hand_landmarks.landmark[hands.HandLandmark.PINKY_TIP].y * image_height))
            for i in range(0, 20):
                sub_data = []
                sub_data.append(str(idx))
                sub_data.append(gesture)
                sub_data.append(joints[i])
                sub_data.append(video_idx)
                sub_data.append(joints_corrdinate[i][1])
                sub_data.append(image_width - joints_corrdinate[i][0])
                data.append(sub_data)
        res = pd.DataFrame(columns=name, data=data)
    return res


def process_videos(filepath, gesture, outputfile):
    filenames = os.listdir(filepath)
    filenames.sort(key=lambda info: int(info[6:-4]))

    video_idx = 0
    dfs = pandas.DataFrame()
    for idx in range(0, len(filenames)):
        video = filepath + '/' + filenames[idx]
        df = detect_hand(video, gesture, video_idx)
        dfs = pd.concat([dfs, df], axis=0, ignore_index=True)
    dfs.to_csv(outputfile)


if __name__ == '__main__':
    process_videos('/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/course_dataset/ASL_letter_A/videos',
                   'ASL_letter_A',
                   '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_1/ASL_letter_A/annotation.csv')

    process_videos('/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/course_dataset/ASL_letter_B/videos',
                   'ASL_letter_B',
                   '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_1/ASL_letter_B/annotation.csv')

    process_videos('/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/course_dataset/ASL_letter_C/videos',
                   'ASL_letter_C',
                   '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_1/ASL_letter_C/annotation.csv')

    process_videos('/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/course_dataset/ASL_letter_L/videos',
                   'ASL_letter_L',
                   '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_1/ASL_letter_L/annotation.csv')

    process_videos('/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/course_dataset/ASL_letter_R/videos',
                   'ASL_letter_R',
                   '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_1/ASL_letter_R/annotation.csv')

    process_videos('/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/course_dataset/ASL_letter_U/videos',
                   'ASL_letter_U',
                   '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_1/ASL_letter_U/annotation.csv')
