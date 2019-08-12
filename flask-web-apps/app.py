from flask import Flask

UPLOAD_FOLDER = "/Users/matthewlee/Desktop/Metis/voice_emotion_classifier/flask-web-apps/upload"

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
