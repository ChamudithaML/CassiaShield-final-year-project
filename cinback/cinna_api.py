# importing required python libraries
import os
from flask import Flask, request, render_template, send_from_directory, Response
from cinback import app
from cinback.img_classify import prediction_func
from cinback.img_classify_ensemble import prediction_func2
from cinback.img_classify_dense import prediction_func3
from cinback.img_classify_vgg import prediction_func4
from cinback.img_classify_inception import prediction_func5
from cinback.camera import Video

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

treatments =["නුහුරු පෑහීම, කොළ කඩා ඉවත් කිරීම, සංස්ථානික කෘමි නාශක යෙදීම. ඇබමසිටින් මිලි ලීටර 6ක් ජලය ලීටර 10ක් සමග එක් කර ඉසින්න ",
             "නීරෝගී පත්‍රයකි ",
             "මැග්නීසියම් ඌනතාවක් නිසා ඇතිවේ. ඩොලමිටේ, කීසරයිට් යෙදීම මගින් වලක්වා ගත හැකිය "]

# route to home page
@app.route('/home')
@app.route('/')
def home_page():
    return render_template('home.html')

# to get the path of directory
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# route to condition checking page
@app.route("/disease1")
def classify_disease1():
    print(APP_ROOT)
    return render_template("disease1.html")

# route to show output result
@app.route("/disease_image", methods=["POST"])
def disease_image():
    # creating path to save image
    target = os.path.join(APP_ROOT, 'images/')

    # creating directory if its not there
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))

    # getting the file from html form
    for im in request.files.getlist("file"):
        diseases = ['Lower Mite Galls', 'Healthy', 'Yellow Cholorosis']
        # getting the file name
        filename = im.filename

        # joining path and file name to variable
        destination = "/".join([target, filename])

        # saving file in destination
        im.save(destination)

        # calling the ML model function to get the prediction
        pred = prediction_func(filename)
        treat=""
        
        if(pred==diseases[0]):
            treat = treatments[0]
        elif(pred==diseases[1]):
            treat=treatments[1]
        else:
            treat = treatments[2]

    return render_template("output1.html", image_name=filename, text=pred, treat=treat)

# route to load image to page
@app.route('/disease_image/<filename>')
def get_cinnamon_img(filename):
    return send_from_directory("images", filename)

# -----------------------------------

# route to login page
@app.route('/login')
def login_page():
    return render_template('login.html')

# route to image gallery page
@app.route('/imageGallery')
def imageGallery_page():
    return render_template('imageGallery.html')

# route to cinnamon guide page
@app.route('/Guide')
def guide_page():
    return render_template('guide1.html')

# ------------------------------------------------------------------

@app.route('/disease_in_real')
def disease_in_real():
    return render_template('disease_in_real.html')

@app.route('/video')
def video():
    return Response(gen(Video()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def gen(obj):
    img_text = ''
    image_x, image_y = 500, 500
    while True:
        frame = obj.get_frame(img_text, image_x, image_y)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')





# References

# https://www.youtube.com/watch?v=qFJeN9V1ZsI
# https://www.youtube.com/watch?v=nmU-FkXr47w
# https://chat.openai.com/
# https://github.com/rrupeshh/Simple-Sign-Language-Detector
# https://www.youtube.com/watch?v=Qr4QMBUPxWo
# https://www.youtube.com/watch?v=C_JKIlc_wlU