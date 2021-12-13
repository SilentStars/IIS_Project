from tensorflow.keras.models import load_model
import numpy as np
import cv2

class LandmarkDetector:
    
    def __init__(self,path_model:str) -> None:
        
        try:
            self.model = load_model(path_model)
        except ImportError as e:
            print(e)
            raise(e)
        except IOError as e:
            print(e)
            raise(e)

        self.input_shape = self.model.input_shape[1:]

    def draw_landmarks(self,frame : np.array,y : np.array, color ='blue') -> np.array:
        image = frame.copy()
        y = y.reshape(20,2)
        radius = 1
        
        if color=='red': #BGR
            color = (255, 0, 0)
        elif color=='blue':
            color = (0,0,255)
        else:
            color = (0,0,0)

        color_border = (255,255,255)
        thickness_circle = -1
        thickness_line = 2

        for landmark in range(20):
            image = cv2.circle(image, tuple(np.array(y[landmark,],int)), radius, color, thickness_circle)

        return image

    def process_frame(self,frame,is_BGR=False):

        if is_BGR:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)             

        resized = cv2.resize(frame,(self.input_shape[1],self.input_shape[0]),interpolation=cv2.INTER_AREA)
        return resized

    def predict(self,formated_img,**kwargs):
        
        # check the input shape
        assert formated_img.shape == self.input_shape, f"input shape {formated_img.shape} expected shape {self.input_shape}"

        # add a dimension
        img = np.expand_dims(formated_img,0)

        return self.model.predict(img,**kwargs)
