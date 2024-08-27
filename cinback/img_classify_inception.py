# Importing required python libraries
import numpy as np    # numpy for arrays
from tensorflow.keras.preprocessing import image   
from tensorflow.keras.models import load_model    # importing keras load_model to import model

# names of classes. use to assign names to prediction
classes = ['Lower Mite Galls', 'Healthy', 'Yellow Cholorosis']

# model using function. taking file name as parameter
def prediction_func5(filename):
    new_model = load_model('cinback/models/inception.h5')   # loading the trained model
    test_image = image.load_img('cinback/images\\' + filename, target_size=(224, 224))   # loading images using file name and resizing to feed the model.
    test_image = image.img_to_array(test_image)   # adding image to numpy array
    test_image = np.expand_dims(test_image, axis=0)   # expanding a dimention to feed the model
    test_image /= 255.
    
    result = new_model.predict(test_image)   # feeding the model
    
    print(result)

    # accessing the prediction result
    result1 = result[0]  
    
    
    
    i = np.argmax(result1)
    
    prediction=""
    
    if(i==0):
        prediction=classes[0]
    elif(i==1):
        prediction = classes[1]
    elif(i==2):
        prediction = classes[2]

    
    return prediction
