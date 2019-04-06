# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 22:14:22 2019

@author: suresh.mani
"""

#Usage: python app.py
import os
 
#os.chdir(r'D:\Projects\Maha Forest\Demo')

from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.preprocessing import image
from keras.models import Sequential, load_model
import numpy as np
import argparse
import tensorflow as tf
#import imutils
#import cv2
import time
import uuid
import base64

img_width, img_height = 150, 150
#model_path = 'D:\Projects\Maha Forest\Demo\Model_v1.h5'
model_path = 'Model_v1.h5'
#model_weights_path = './models/weights.h5'
model = load_model(model_path)
model._make_predict_function()
graph = tf.get_default_graph()
#model.load_weights(model_weights_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'JPG'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predict(file):
    img = image.load_img(file, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    global graph
    with graph.as_default():
        prediction = model.predict(img_tensor)
    #result = array[0]
    #answer = np.argmax(result)
    '''
    if answer > 0.5:
        print("There is an Encroachment in this location")
    else:
	    print("There is no Encroachment in this location")
    '''
    return prediction

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', label='', imagesource='../uploads/Mahaforestdepartment.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            #print(result)
            if result > 0.5:
                label = '"There is an Encroachment in this location"'
            else:
                label = 'There is no Encroachment in this location'			
            print(result)
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            print("--- %s Probability ---" % str (result))
            return render_template('template.html', label=label, imagesource='../uploads/' + filename)

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file', build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})
    
if __name__ == "__main__":
    app.debug=False
    app.run(host='0.0.0.0', port=3000)

    
    
    
