import numpy as np
import matplotlib.pyplot as plt
import pylab
import cv2


def read_record(path):
    """

    Args:
        path: The path of the video file.

    Returns: An array containing all the frames in the video.

    """
    cap = cv2.VideoCapture(path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        else:
            frames.append(frame)
    return frames


read_record('/Users/houqinhan/Desktop/UppsalaCourse/IIS/IIS_Project/sub_system_2/course_dataset/ASL_letter_A/videos/video_0.mp4')