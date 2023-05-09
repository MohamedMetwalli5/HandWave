from flask import Flask, render_template, request, session, logging, url_for, redirect, flash, Response, send_file
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sqlalchemy.orm import scoped_session, sessionmaker

import cv2
import numpy as np

import tensorflow as tf
from object_detection.utils import config_util
import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import sys

from gtts import gTTS
from IPython.display import Audio
sys.path.append("C:/Users/Aya/Handwave/Tensorflow/models")
sys.path
from object_detection.builders import model_builder


from passlib.hash import sha256_crypt

engine = create_engine("mysql+pymysql://root:1234567@localhost/register")


MAIN_PATH = 'C:/Users/Aya/Handwave'
WORKSPACE_PATH = MAIN_PATH + '/Tensorflow/workspace'
SCRIPTS_PATH = MAIN_PATH + '/Tensorflow/scripts'
APIMODEL_PATH = MAIN_PATH + '/Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
IMAGE_PATH = WORKSPACE_PATH + '/images'
MODEL_PATH = WORKSPACE_PATH + '/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH  + '/pre-trained-models'
CONFIG_PATH = MODEL_PATH + '/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH + '/my_ssd_mobnet/'

camera = cv2.VideoCapture(0)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

db = scoped_session(sessionmaker(bind=engine))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    if 'log' in session:
        return render_template("home.html")
    else:
        flash("You need to login first!", "danger")
        return redirect(url_for('login'))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        username = request.form.get("username")
        password = request.form.get("passIn")
        confirm = request.form.get("passConf")
        securePass = sha256_crypt.encrypt(str(password))
        
        usernamedata = db.execute(text("SELECT users.username FROM users WHERE users.username=:username"), {"username":username}).fetchone()
        if not (usernamedata is None):
            flash("This username already in use, try another one!", "danger")
            return render_template("register.html")
        elif password == confirm:
            db.execute(text("INSERT INTO users(name, username, password) VALUES(:name, :username, :password)"), {"name" :name, "username" :username, "password" :securePass})
            db.commit()
            flash("your account is created successfully, you can login now", "success")
            return redirect(url_for('login'))
        else:
            flash("Confirmation password isn't matched!", "danger")
            return render_template("register.html")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("usernameLog")
        password = request.form.get("passLog")

        usernamedata = db.execute(text("SELECT users.username FROM users WHERE users.username=:username"), {"username":username}).fetchone()
        if usernamedata is None:
            flash("Wrong Username!", "danger")
            return render_template("login.html")
        else:
            passworddata = db.execute(text("SELECT users.password FROM users WHERE users.username=:username"), {"username":username}).fetchone()
            for passwrd in passworddata:
                if not (sha256_crypt.verify(password, passwrd)):
                    flash("Wrong Password!", "danger")
                    return render_template("login.html")
                else:
                    session["log"] = True
                    return redirect(url_for('home'))
    else:
        if 'log' in session:
            return redirect(url_for('home'))
        else:
            return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/home/speak", methods=["GET", "POST"])
def speak():
    if request.method == "POST":
        str = request.form.get("str")
        tts = gTTS(str)
        tts.save('1.wav')
        sound_file = '1.wav'
        Audio(sound_file, autoplay=True)
        
def generate_frames():
    camera = cv2.VideoCapture(0)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-21')).expect_partial()

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')
    

    while True:
        ret, frame = camera.read()
        if not ret:
            break
        image_np = np.array(frame)
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        label_id_offset = 1
        image_np_with_detections = image_np.copy()
        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=2,
                    min_score_thresh=.95,
                    agnostic_mode=False)
        
        ret, buffer = cv2.imencode('.jpg', cv2.resize(image_np_with_detections, (800, 600)))
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.secret_key = "1234567"
    app.run(debug = True)
