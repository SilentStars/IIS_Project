import cv2
import mediapipe as mp
import func
import pandas


def setting_mediapipe():
    """

    Run this function to set up mediapipe library.

    """
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    return mp_drawing, mp_drawing_styles, mp_hands


def detect_hand(video, ):



frames = func.read_record(
    '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/course_dataset/ASL_letter_A/videos/video_0.mp4')
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
) as hands:
    for idx, file in enumerate(frames):
        image = cv2.flip(file, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print('Handedness:', results.multi_handedness)
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
        cv2.imwrite(
            '/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_1/tmp/annotated_image' + str(
                idx) + '.png', cv2.flip(annotated_image, 1))
