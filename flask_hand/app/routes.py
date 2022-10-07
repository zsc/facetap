import sys
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import random
import os
import cv2
import glob
import numpy as np
import Jetson.GPIO as gpio
try:
   import cPickle as pickle
except:
   import pickle

from flask import make_response, render_template, Response
from app import app

cap = None
continuous_frames = 0
threshold_on = 6
threshold_off = 3

def turn_on():
    gpio.setup(7, gpio.OUT)

def turn_off():
    #gpio.setup(17, gpio.OUT)
    gpio.cleanup(7)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

def gen():
    while True:
        frame = get_frame()
        global g_image
        g_image = frame
        _, img_encoded = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_encoded.tostring() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def get_frame():
    global cap
    global faceCascade
    if cap is None:
        dev = '/dev/video0'
        if not os.path.exists(dev):
            dev = 0
        cap = cv2.VideoCapture(dev)
        gpio.setmode(gpio.BOARD)

    try:
        ret, img = cap.read()
        scale = 224 
        y0 = (img.shape[0] - scale)//2
        x0 = (img.shape[1] - scale)//2
        img = img[y0:y0+scale, x0:x0+scale]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)


        '''
        global continuous_frames
        if len(faces) > 0:
            continuous_frames += 1
        else:
            continuous_frames = 0

        #global threshold_on
        #global threshold_off
        if continuous_frames > 6:
            turn_on()
        elif continuous_frames < 3:
            turn_off()

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        '''

    except Exception as e:
        print(e)
        img = np.zeros((40, 40, 3)).astype('uint8')
    return cv2.resize(img, (224, 224))

def proc_cmd(cmd):
    print(cmd)

@app.route('/on')
def on():
    proc_cmd('on')
    return render_template('index.html')

@app.route('/off')
def off():
    proc_cmd('off')
    return render_template('index.html')

@app.route('/threshold_on')
def threshold_on(val):
    global threshold_on
    threshold_on = val
    return render_template('index.html')

@app.route('/threshold_off')
def threshold_off(val):
    global threshold_off
    threshold_off = val
    return render_template('index.html')

@app.route('/reboot')
def reboot():
    print('reboot')
    os.system('yes ubuntu|sudo reboot')
    return ""

