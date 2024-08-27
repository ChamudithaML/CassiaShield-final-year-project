# Importing required python libraries
import cv2 as cv
import tensorflow as tf
import numpy as np  # numpy for arrays
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model  # importing keras load_model to import model


class Video(object):
    def __init__(self):
        # self.video = cv.VideoCapture(0, cv.CAP_DSHOW)
        # Setting the camera to use
        self.video = cv.VideoCapture(0)

        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in self.gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(self.gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.classifier = load_model('cinback/models/cinna3.h5')
        self.classifier2 = load_model('cinback/models/CinaMiNet2.h5')
        self.classes = ['Lower Mite Galls', 'Healthy', 'Yellow Cholorosis']

    def predictor(self):
        test_image = image.load_img('cinback/1.png', target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        result = self.classifier.predict(test_image)

        # accessing the prediction result
        result1 = result[0]

        # variable to hold array index
        k = 0

        # this loop to assign class name according to output
        for i in range(3):
            if result1[i] == 1.:
                k = i
                break

        # taking class name using index and storing it in variable
        prediction = self.classes[k]

        # returning class name
        return prediction
    
    def predictor2(self):
        test_image = image.load_img('cinback/1.png', target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image /= 255.

        result = self.classifier2.predict(test_image)
        
        result1 = result[0]  
    
        i = np.argmax(result1)
        
        prediction=""
        
        if(i==0):
            prediction = self.classes[0]
        elif(i==1):
            prediction = self.classes[1]
        elif(i==2):
            prediction = self.classes[2]

        # returning class name
        return prediction

    def get_frame(self, img_text, image_x, image_y):
        # this code will turn on the webcam and get the video feed into the system
        ret, frame = self.video.read()
        # flipping the frame to make easier to the user
        frame = cv.flip(frame, 1)

        # ----------------------------------------------------------------------------
        # Drawing the bounding box
        img = cv.rectangle(frame, (150, 100), (450, 400), (0, 255, 0), thickness=2, lineType=8, shift=0)
        # Cropping the bounding box area
        imcrop = img[102:398, 152:448]
        # Taking the prediction as string from model
        img_text = self.predictor2()
        # Putting the predicted text to the frame image
        cv.putText(frame, img_text, (30, 450), cv.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 255))

        # cv.imshow("test", frame)
        # Image needs to be saved to feed the model. giving a file name to save the image
        img_name = "cinback/1.png"
        # resizing the cropped image to make it bigger before saving the image
        save_img = cv.resize(imcrop, (image_x, image_y))
        # saving the image. image will be fed to the model in next iteration
        cv.imwrite(img_name, save_img)

        # print("{} written!".format(img_name))

        # -----------------------------------------------------------------------

        ret, jpg = cv.imencode('.jpg', frame)
        return jpg.tobytes()