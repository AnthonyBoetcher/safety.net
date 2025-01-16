from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")
model_custom = YOLO('best.pt')  # Update with the correct path to your YOLOv8 model

@app.route('/')
def index():
    return render_template('index.html')  # Serves the HTML file

@socketio.on('message')
def handle_frame(data):
    # Decode the base64 image
    header, encoded = data.split(',', 1)
    frame = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    # YOLO inference
    results = model_custom.predict(frame)

    # Draw bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            if confidence > 0.5:
                color = (0, 255, 0)  # Customize colors here
                label = f"{model_custom.names[class_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Encode frame back to base64 to send it to the client
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_frame = base64.b64encode(buffer).decode('utf-8')
    socketio.emit('response', f"data:image/jpeg;base64,{encoded_frame}")

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
