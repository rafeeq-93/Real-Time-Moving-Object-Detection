from flask import Flask, render_template, Response
import cv2
import time
import imutils

app = Flask(__name__)

# Initialize camera
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    raise Exception("Error: Could not access the camera.")

time.sleep(1)

firstFrame = None  # Store the initial frame for motion detection
area = 500  # Minimum area for detecting motion


def generate_frames():
    global firstFrame

    while True:
        success, img = cam.read()
        if not success:
            break

        text = "Normal"
        img = imutils.resize(img, width=1000)

        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)

        if firstFrame is None:
            firstFrame = gaussianImg
            continue

        imgDiff = cv2.absdiff(firstFrame, gaussianImg)
        threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]
        threshImg = cv2.dilate(threshImg, None, iterations=2)

        cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            if cv2.contourArea(c) < area:
                continue
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = "Moving obj detected"

        cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', img)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
