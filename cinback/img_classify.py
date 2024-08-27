# Importing required python libraries
import numpy as np    # numpy for arrays
from tensorflow.keras.preprocessing import image   
from tensorflow.keras.models import load_model    # importing keras load_model to import model

# names of classes. use to assign names to prediction
classes = ['Lower Mite Galls', 'Healthy', 'Yellow Cholorosis']

# model using function. taking file name as parameter
def prediction_func(filename):
    new_model = load_model('cinback/models/CinaMiNet2.h5')   # loading the trained model
    test_image = image.load_img('cinback/images\\' + filename, target_size=(150, 150))   # loading images using file name and resizing to feed the model.
    test_image = image.img_to_array(test_image)   # adding image to numpy array
    test_image = np.expand_dims(test_image, axis=0)   # expanding a dimention to feed the model
    test_image /= 255.  # comment out if using cinna3
    result = new_model.predict(test_image)   # feeding the model

    # accessing the prediction result
    # result1 = result[0]  # need for cinna3
    print("XXXXXXXXXXXXXXXXXXXX")
    print(result)
    print("XXXXXXXXXXXXXXXXXXX")
    
    # ---------------------------------------------------------------
    # variable to hold array index
    # k=0 

    # this loop to assign class name according to output
    # for i in range(3):
    #     if result1[i] == 1.:
    #         k=i
    #         break

    # taking class name using index and stroring it in variable
    # prediction = classes[k]

    # returning class name
    #----------------------------------------------------------------
    
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
