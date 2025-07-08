import argparse
import io
from PIL import Image
import datetime
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import time
from ultralytics import YOLO
import pywhatkit

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            basepath = os.path.dirname(__file__)
            filepath = os.path.join(basepath, 'uploads', f.filename)
            print("upload folder is ", filepath)
            f.save(filepath)

            # Specify the full path for the output video
            output_path = os.path.join(basepath, 'output.mp4')

            file_extension = f.filename.rsplit('.', 1)[1].lower()

            if file_extension == 'jpg':
                img = cv2.imread(filepath)
                frame = cv2.imencode('.jpg', cv2.UMat(img))[1].tobytes()
                image = Image.open(io.BytesIO(frame))

                # Perform the detection
                model = YOLO('best.pt')
                detections = model(image, show=True, plots=True)
                return "error"

            elif file_extension == 'mp4':
                video_path = filepath
                cap = cv2.VideoCapture(video_path)

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Define the codec and create VideoWriter with the full path
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

                model = YOLO('best.pt')

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, save=True, plots=True, conf=0.8)
                    print(results)
                    for r in results:
                        print(len(r.boxes))
                        if len(r.boxes) > 0:
                            print("Accident is present")
                        else:
                            print("No accident detected")
                    cv2.waitKey(1)

                    res_plotted = results[0].plot()
                    cv2.imshow("result", res_plotted)

                    # write the processed frame to output video
                    out.write(res_plotted)

                    if cv2.waitKey(1) == ord('q'):
                        break
                
                cap.release()
                out.release()
                cv2.destroyAllWindows()

                return video_feed()

    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    image_path = folder_path + '/' + latest_subfolder + '/' + f.filename
    return render_template('index.html', image_path=image_path)

@app.route('/<path:filename>')
def display(filename):
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    directory = folder_path + '/' + latest_subfolder
    print("printing directory: ", directory)
    files = os.listdir(directory)
    latest_file = files[0]

    print(latest_file)

    filename = os.path.join(folder_path, latest_subfolder, latest_file)

    file_extension = filename.rsplit('.', 1)[1].lower()

    environ = request.environ
    if file_extension == 'jpg':
        return send_from_directory(directory, latest_file, environ)

    else:
        return "Format file salah!"

def get_frame():
    # Path to the saved output video
    mp4_files = 'output.mp4'
    video = cv2.VideoCapture(mp4_files)  # detected video path
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)  # control the frame rate to display one frame every 100 milliseconds:

@app.route("/video_feed")
def video_feed():
    print("function called")

    # Update sendwhatmsg to include the time arguments
    time_hour = datetime.datetime.now().hour
    time_min = (datetime.datetime.now().minute + 2) % 60  # send after 2 minutes

    # pywhatkit.sendwhatmsg("+966553977714", "Accident is present", time_hour, time_min)

    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# New function: Save the frame with the highest confidence
def save_highest_conf_frame(video_path, output_image_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO('best.pt')
    highest_conf = 0
    highest_conf_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for r in results:
            for box in r.boxes:
                conf = box.conf.item()
                if conf > highest_conf:
                    highest_conf = conf
                    highest_conf_frame = frame.copy()

    cap.release()

    if highest_conf_frame is not None:
        # Save the frame with the highest confidence
        cv2.imwrite(output_image_path, highest_conf_frame)
        print(f"Saved highest confidence frame to {output_image_path}")
    else:
        print("No detections with confidence found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov8 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)
