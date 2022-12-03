import sys

# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import random
import os
import cv2
import PIL
import glob
import numpy as np
import onnxruntime as ort
import Jetson.GPIO as gpio

try:
    import cPickle as pickle
except:
    import pickle

from flask import make_response, render_template, Response
from app import app

print(ort.get_device())
ort_session = ort.InferenceSession("/home/dev/1113_model/model.onnx")
classes = [
    "good",
    "bad",
    "no_hand",
]
cap = None
continuous_frames = 0
# threshold_on = 6
# threshold_off = 3
do_capture = False
g_frame_cnt = 0
g_pred_word = "UNK"
g_pred_cls = ""
g_image = None

gpio.setmode(gpio.BOARD)
gpio.setup(7, gpio.OUT)


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html", title="{}".format(g_pred_cls))


def gen():
    while True:
        frame = get_frame()
        _, img_encoded = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + img_encoded.tostring() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def refresh_pred():
    img = g_image[:]
    if img is None:
        return
    inp_numpy = cv2.resize(img, (300, 300))[:, :, ::-1][None].astype("float32")
    class_scores = ort_session.run(None, {"input": inp_numpy})[0][0]

    global g_pred_word
    g_pred_word = "{} {}".format(classes[class_scores.argmax()], class_scores.max())
    global g_pred_cls
    if class_scores.max() < 0.7:
        g_pred_cls = "unk"
    else:
        g_pred_cls = classes[class_scores.argmax()]


def get_frame():
    global cap
    global faceCascade
    if cap is None:
        dev = "/dev/video0"
        if not os.path.exists(dev):
            dev = 0
        cap = cv2.VideoCapture(dev)

    try:
        ret, img = cap.read()
        if img is not None:
            # print(img.shape)
            # print(img.transpose(2, 0, 1).shape)
            img = cv2.resize(img, (320, 240))
            scale = 224
            y0 = (img.shape[0] - scale) // 2
            x0 = (img.shape[1] - scale) // 2
            img = img[y0 : y0 + scale, x0 : x0 + scale]
            global g_image
            g_image = img[:]

            global g_frame_cnt
            g_frame_cnt += 1
            if g_frame_cnt % 10 == 0 and do_capture:
                cv2.imwrite("/home/dev/captures/{:05d}.jpg".format(g_frame_cnt), g_image)

            refresh_pred()
            cv2.putText(
                img,
                g_pred_word,
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            # print(g_pred_word)

        else:
            print(ret, img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)

        """
        global continuous_frames
        if len(faces) > 0:
            continuous_frames += 1
        else:
            continuous_frames = 0

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        """

    except Exception as e:
        print(e)
        img = np.zeros((40, 40, 3)).astype("uint8")
    return img  # cv2.resize(img, (224, 224))


def proc_cmd(cmd):
    print(cmd)


@app.route("/status")
def status():
    refresh_pred()
    return g_pred_cls


@app.route("/on")
def on():
    proc_cmd("on")
    return render_template("index.html", title="{}".format(g_pred_cls))


@app.route("/off")
def off():
    proc_cmd("off")
    return render_template("index.html", title="{}".format(g_pred_cls))


@app.route("/threshold_on")
def threshold_on():
    # global threshold_on
    # threshold_on = val
    refresh_pred()
    gpio.output(7, True)
    return render_template("index.html", title="{}".format(g_pred_cls))


@app.route("/threshold_off")
def threshold_off():
    # global threshold_off
    # threshold_off = val
    refresh_pred()
    gpio.output(7, False)
    return render_template("index.html", title="{}".format(g_pred_cls))


@app.route("/reboot")
def reboot():
    print("reboot")
    os.system("yes ubuntu|sudo reboot")
    return ""
