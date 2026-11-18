import os
os.environ["GLOG_minloglevel"] = "2"        # hide MediaPipe/TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import serial
except ImportError:
    serial = None

# ---------------- settings ----------------
CAMERA_INDEX = 1              # USB webcam (tried first)
LAPTOP_CAMERA_INDEX = 0       # never used
SERIAL_PORT = "COM8"
BAUD_RATE = 115200
FACE_MODEL = "blaze_face_short_range.tflite"

PAN_CENTER = 90
TILT_CENTER = 90
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 30, 150
PAN_INVERT = False            # press X live to flip
TILT_INVERT = True            # press Y live to flip

CAMERA_FOV_X = 60             # webcam horizontal view angle in degrees
CAMERA_FOV_Y = 45             # webcam vertical view angle in degrees
TRACK_SPEED = 2.5             # how fast to close the gap (per second), +/- keys change it
MAX_SPEED = 90                # degrees per second limit
DEADZONE = 0.06               # fraction of frame: centre area where it doesn't move
FACE_SMOOTH = 0.5             # 0 = raw, 0.8 = very smooth
SEND_INTERVAL = 0.03
HEARTBEAT = 0.2               # resend even if unchanged so ESP knows Python is alive
# ------------------------------------------


def open_serial():
    if serial is None:
        print("pyserial not installed -> pip install pyserial")
        return None
    try:
        s = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        time.sleep(2)
        print("Connected to ESP32 on", SERIAL_PORT)
        return s
    except Exception as e:
        print("No ESP32 on", SERIAL_PORT, "-", e)
        return None


def send(cmd):
    if ser is not None:
        try:
            ser.write((cmd + "\n").encode())
        except Exception as e:
            print("Serial error:", e)


def open_camera(preferred):
    order = [preferred] + [i for i in range(1, 5) if i != preferred]
    order = [i for i in order if i != LAPTOP_CAMERA_INDEX]
    for index in order:
        for name, api in [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("default", cv2.CAP_ANY)]:
            c = cv2.VideoCapture(index, api)
            if not c.isOpened():
                c.release()
                continue
            for _ in range(40):
                ok, _ = c.read()
                if ok:
                    print("Camera", index, "opened with", name)
                    return c
                time.sleep(0.05)
            c.release()
    return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


ser = open_serial()

options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.4,
)
detector = vision.FaceDetector.create_from_options(options)

cap = open_camera(CAMERA_INDEX)
if cap is None:
    print("USB webcam not found. Close the notebook kernel / other apps using it, replug, retry.")
    raise SystemExit

start_time = time.time()
last_frame_time = time.time()

pan = float(PAN_CENTER)
tilt = float(TILT_CENTER)
face_cx = None
face_cy = None
last_sent = None
last_send_time = 0.0
tracking = True
send("CAM:%d,%d" % (PAN_CENTER, TILT_CENTER))

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped sending frames")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        now = time.time()
        dt = min(now - last_frame_time, 0.1)
        last_frame_time = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int((now - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp)

        if result.detections:
            best = max(result.detections,
                       key=lambda d: d.bounding_box.width * d.bounding_box.height)
            box = best.bounding_box
            cx = box.origin_x + box.width / 2
            cy = box.origin_y + box.height / 2
            if face_cx is None:
                face_cx, face_cy = cx, cy
            else:
                face_cx = FACE_SMOOTH * face_cx + (1 - FACE_SMOOTH) * cx
                face_cy = FACE_SMOOTH * face_cy + (1 - FACE_SMOOTH) * cy

            err_x = face_cx / width - 0.5     # -0.5 (left edge) .. +0.5 (right edge)
            err_y = face_cy / height - 0.5

            if tracking:
                # how many degrees the face is away from the centre
                off_x = err_x * CAMERA_FOV_X if abs(err_x) > DEADZONE else 0.0
                off_y = err_y * CAMERA_FOV_Y if abs(err_y) > DEADZONE else 0.0
                move_x = clamp(off_x * TRACK_SPEED * dt, -MAX_SPEED * dt, MAX_SPEED * dt)
                move_y = clamp(off_y * TRACK_SPEED * dt, -MAX_SPEED * dt, MAX_SPEED * dt)
                pan = clamp(pan + (-move_x if PAN_INVERT else move_x), PAN_MIN, PAN_MAX)
                tilt = clamp(tilt + (-move_y if TILT_INVERT else move_y), TILT_MIN, TILT_MAX)

            cv2.rectangle(frame, (box.origin_x, box.origin_y),
                          (box.origin_x + box.width, box.origin_y + box.height), (255, 0, 255), 2)
            cv2.line(frame, (width // 2, height // 2), (int(face_cx), int(face_cy)), (255, 0, 255), 1)
            cv2.putText(frame, "err x %+.2f  y %+.2f" % (err_x, err_y), (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        else:
            face_cx = None
            face_cy = None
            cv2.putText(frame, "No face (holding position)", (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cam = (round(pan, 1), round(tilt, 1))
        changed = cam != last_sent and now - last_send_time >= SEND_INTERVAL
        if changed or now - last_send_time >= HEARTBEAT:
            send("CAM:%.1f,%.1f" % cam)
            last_sent = cam
            last_send_time = now

        # dead zone box + centre cross
        dz_w, dz_h = int(DEADZONE * width), int(DEADZONE * height)
        cv2.rectangle(frame, (width // 2 - dz_w, height // 2 - dz_h),
                      (width // 2 + dz_w, height // 2 + dz_h), (255, 255, 255), 1)
        cv2.drawMarker(frame, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

        cv2.putText(frame, "pan %.0f  tilt %.0f  tracking %s  speed %.1f"
                    % (pan, tilt, "ON" if tracking else "OFF", TRACK_SPEED),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "invert X:%s Y:%s" % (PAN_INVERT, TILT_INVERT), (10, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, "ESP32: " + ("connected" if ser else "offline")
                    + "  F=track C=centre X/Y=flip +/-=speed Q=quit",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Face follow test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            tracking = not tracking
        elif key == ord("c"):
            pan, tilt = float(PAN_CENTER), float(TILT_CENTER)
        elif key == ord("x"):
            PAN_INVERT = not PAN_INVERT
        elif key == ord("y"):
            TILT_INVERT = not TILT_INVERT
        elif key in (ord("+"), ord("=")):
            TRACK_SPEED = min(8.0, TRACK_SPEED + 0.5)
        elif key == ord("-"):
            TRACK_SPEED = max(0.5, TRACK_SPEED - 0.5)
finally:
    send("CAM:%d,%d" % (PAN_CENTER, TILT_CENTER))
    time.sleep(0.1)
    if ser is not None:
        ser.close()
    cap.release()
    cv2.destroyAllWindows()