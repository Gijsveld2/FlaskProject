from flask import Flask, render_template, request, Response, send_file
from ultralytics import YOLO
import os
import cv2
import threading
import time
import json
from datetime import datetime
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'
app.config['REPORTS_FOLDER'] = 'static/reports'
app.config['HISTORY_FILE'] = 'history.json'

model = YOLO("yolov8n.pt", verbose=False)

CLASS_NAMES = {
    47: "apple",
    46: "banana",
    49: "orange"
}

current_frame = None
is_processing = False
report_data = []
current_video = None


def load_history():
    if not os.path.exists(app.config['HISTORY_FILE']):
        return []
    try:
        with open(app.config['HISTORY_FILE'], 'r') as f:
            return json.load(f)
    except:
        return []


def save_history(history):
    with open(app.config['HISTORY_FILE'], 'w') as f:
        json.dump(history, f, indent=4)


def add_to_history(file_type, filename, detections):
    history = load_history()
    history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_type": file_type,
        "filename": filename,
        "detections": detections
    })
    save_history(history)


def save_processed_image(image_path, annotated_frame):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(image_path))
    cv2.imwrite(processed_path, annotated_frame)
    return processed_path


def save_processed_video(video_path, frames, fps, size):
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_path, fourcc, fps, size)

    for frame in frames:
        out.write(frame)
    out.release()
    return processed_path


def process_video(video_path):
    global current_frame, is_processing, report_data, current_video

    try:
        current_video = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Не удалось открыть видео")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        report_data = []
        is_processing = True
        processed_frames = []

        while cap.isOpened() and is_processing:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            results = model(frame)
            annotated_frame = results[0].plot()
            processed_frames.append(annotated_frame)

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            current_frame = buffer.tobytes()

            detections = results[0].boxes.cls.tolist()
            counts = {"apple": 0, "banana": 0, "orange": 0}
            for cls in detections:
                if cls in CLASS_NAMES:
                    counts[CLASS_NAMES[cls]] += 1

            report_data.append({
                "time_sec": round(current_time, 2),
                "total_objects": len(detections),
                "objects": counts.copy()
            })

            time.sleep(0.03)

        cap.release()

        if report_data:
            try:
                generate_excel_report(report_data, current_video)
                processed_video_path = save_processed_video(video_path, processed_frames, fps, (width, height))
                add_to_history("video", os.path.basename(processed_video_path), report_data)
            except Exception as e:
                print(f"Ошибка при генерации отчёта: {str(e)}")

    except Exception as e:
        print(f"Ошибка обработки видео: {str(e)}")
    finally:
        is_processing = False
        current_frame = None


def generate_excel_report(data, video_name):
    try:
        df = pd.DataFrame(data)
        df = pd.concat([
            df.drop(['objects'], axis=1),
            pd.json_normalize(df['objects'])
        ], axis=1)

        os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
        report_path = os.path.join(app.config['REPORTS_FOLDER'], f"report_{video_name}.xlsx")
        df.to_excel(report_path, index=False, engine='openpyxl')
        return report_path
    except Exception as e:
        raise Exception(f"Excel generation error: {str(e)}")


@app.route("/")
def index():
    history = load_history()
    return render_template("index.html", history=history, is_processing=is_processing)


@app.route("/video_feed")
def video_feed():
    def generate():
        global current_frame
        while is_processing or current_frame:
            if current_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
            time.sleep(0.03)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/upload", methods=["POST"])
def upload():
    global is_processing

    if "file" not in request.files:
        return "Файл не загружен", 400

    file = request.files["file"]
    if file.filename == "":
        return "Файл не выбран", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            results = model(filepath)
            annotated_frame = results[0].plot()
            processed_path = save_processed_image(filepath, annotated_frame)

            detections = results[0].boxes.cls.tolist()
            counts = {"apple": 0, "banana": 0, "orange": 0}
            for cls in detections:
                if cls in CLASS_NAMES:
                    counts[CLASS_NAMES[cls]] += 1

            add_to_history("image", os.path.basename(processed_path), counts)

            return render_template(
                "index.html",
                result=counts,
                input_file=filepath,
                processed_file=processed_path,
                history=load_history(),
                is_processing=False
            )
        except Exception as e:
            return f"Ошибка обработки изображения: {str(e)}", 500

    elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
        if is_processing:
            return "Уже идёт обработка видео", 400

        is_processing = True
        threading.Thread(target=process_video, args=(filepath,)).start()

        return render_template(
            "index.html",
            input_file=filepath,
            is_video=True,
            history=load_history(),
            is_processing=True
        )


@app.route("/download_report/<filename>")
def download_report(filename):
    try:
        return send_file(
            os.path.join(app.config['REPORTS_FOLDER'], filename),
            as_attachment=True
        )
    except Exception as e:
        return f"Ошибка загрузки отчёта: {str(e)}", 404


@app.route("/stop_processing")
def stop_processing():
    global is_processing
    is_processing = False
    return "Обработка остановлена", 200


if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)
    app.run(debug=True)