from flask import Flask, render_template, Response,jsonify,request,session,url_for
from werkzeug.utils import secure_filename
import os
from detect import image_detection,video_detection,detection_sign
import cv2
from flask_wtf import FlaskForm
from wtforms.validators import InputRequired,NumberRange
import pyttsx3
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
import threading
import time
import base64


application= Flask(__name__)
application.config['SECRET_KEY'] = 'amrutha'
application.config['UPLOAD_FOLDER'] = 'static/files'
detection_sign=""

class UploadFileForm(FlaskForm):
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")

def generate_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames_web(path_x):
    global detection_sign
    yolo_output = video_detection(path_x) 
    for detection_, detected_sign in yolo_output:  
        ref, buffer = cv2.imencode('.jpg', detection_) 
        frame = buffer.tobytes() 
        detection_sign=detected_sign
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@application.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@application.route('/speak', methods=['POST'])
def speak():
    data = request.get_json()
    text = data.get('text', '')
    print(f"Received text: {text}")
    if text:
        threading.Thread(target=generate_speech, args=(text,)).start()
        return jsonify({"message": "Speech output generated"}), 200

@application.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    filename = secure_filename(file.filename)
    file_path = os.path.join(application.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    predicted_sign, processed_image_path = image_detection(file_path)
    image_url = url_for('static', filename=f'files/{os.path.basename(processed_image_path)}')
    return jsonify({
        'predicted_sign': predicted_sign,
        'image_url': image_url
    })

@application.route("/webcam", methods=['GET', 'POST'])
def webcam():
    session.clear()
    return render_template('livefeed.html')

@application.route('/video')
def video():
    return Response(generate_frames(path_x=session.get('video_path', None)), mimetype='multipart/x-mixed-replace; boundary=frame')

@application.route('/webapplication')
def webapplication():
    return Response(generate_frames_web(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@application.route('/detect_sign', methods=['GET'])
def detect_sign():
    global detection_sign  
    return jsonify({'predicted_sign': detection_sign})

@application.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the image data from request
        image_data = request.json['image']
        # Convert base64 image to cv2 format
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame using your existing detection logic
        predicted_sign = video_detection(img)
        
        return jsonify({
            'success': True,
            'predicted_sign': predicted_sign
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == "__main__":
    application.run(debug=True,use_reloader=False)                  
