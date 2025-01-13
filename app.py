from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)

# Emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load the trained model
emotion_model = load_model('model/emotion_model_full.h5')
emotion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Get the image data from the client
        data = request.json['image']
        image_data = base64.b64decode(data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load the face detector
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

        # Detect faces
        faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        emotions = []

        for (x, y, w, h) in faces:
            # Extract face region
            roi_gray = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            # Predict emotion
            prediction = emotion_model.predict(cropped_img)
            emotion = emotion_dict[np.argmax(prediction)]
            emotions.append(emotion)

        return jsonify({'emotions': emotions})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
