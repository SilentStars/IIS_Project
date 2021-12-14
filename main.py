import cv2
from sub_system_1.LandmarkDetector import LandmarkDetector
from sub_system_2.gestureClassification import GestureClassifier
import sys,os

def get_video(input_id,detector,clf):
    cap = cv2.VideoCapture(input_id)
    print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
    while cap.isOpened():
        okay, frame = cap.read()
        if not okay:
            break

        g,frame = get_gesture(frame,detector,clf)
        print(g)
        cv2.imshow('video', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print('cleaning done')
    pass



def get_gesture(frame,detector,clf):
    processed = detector.process_frame(frame,is_BGR=True)
    coord = detector.predict(processed)
    gest = clf.predict(coord)

    frame = detector.draw_landmarks(frame,coord*4)
    return gest,frame

if __name__ == '__main__':
    if len(sys.argv) == 2:
        f = sys.argv[1]
        assert os.path.exists(f), f"{f} doesn't exist"
        id = f
        print(f"video from {f} will be played")
    elif len(sys.argv) == 1:
        id = 0
        print(f"input stream from webcam")
    else:
        raise Exception(f'{len(sys.argv) - 1} arguments passed. 1 at most is expected')

    detector = LandmarkDetector('sub_system_1/Trained_Model')
    clf = GestureClassifier('sub_system_2/models/mlp','sub_system_2/models/encoder.obj')
    get_video(id,detector,clf)
