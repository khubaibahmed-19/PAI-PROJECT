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
CAMERA_INDEX = 1
SERIAL_PORT = "COM8"          # your ESP32 port (Device Manager -> Ports)
BAUD_RATE = 115200
FACE_MODEL = "blaze_face_short_range.tflite"

PAN_CENTER = 90
TILT_CENTER = 90
PAN_MIN, PAN_MAX = 0, 180
TILT_MIN, TILT_MAX = 30, 150
PAN_INVERT = False            # flip if the camera turns AWAY from your face
TILT_INVERT = False
TRACK_GAIN = 8.0              # degrees per frame when face is at the frame edge
TRACK_DEADZONE = 0.04
FACE_SMOOTH = 0.6
SEND_INTERVAL = 0.03
# ------------------------------------------


def open_serial():
    if serial is None:
        print("pyserial not installed -> run:  %pip install pyserial")
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


ser = open_serial()

options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    min_detection_confidence=0.6,
)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
start_time = time.time()

pan = float(PAN_CENTER)
tilt = float(TILT_CENTER)
face_cx = None
face_cy = None
last_sent = None
last_send_time = 0.0
tracking = True
send("CAM:%d,%d" % (PAN_CENTER, TILT_CENTER))

while True:
    ok, frame = cap.read()
    if not ok:
        print("Camera not working, check CAMERA_INDEX")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    now = time.time()

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

        err_x = face_cx / width - 0.5
        err_y = face_cy / height - 0.5
        if tracking:
            if abs(err_x) > TRACK_DEADZONE:
                pan += (-1 if PAN_INVERT else 1) * TRACK_GAIN * err_x
            if abs(err_y) > TRACK_DEADZONE:
                tilt += (-1 if TILT_INVERT else 1) * TRACK_GAIN * err_y
            pan = max(PAN_MIN, min(PAN_MAX, pan))
            tilt = max(TILT_MIN, min(TILT_MAX, tilt))

        cv2.rectangle(frame, (box.origin_x, box.origin_y),
                      (box.origin_x + box.width, box.origin_y + box.height), (255, 0, 255), 2)
        cv2.line(frame, (width // 2, height // 2), (int(face_cx), int(face_cy)), (255, 0, 255), 1)
        cv2.putText(frame, "err x %.2f  y %.2f" % (err_x, err_y), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        face_cx = None
        face_cy = None
        cv2.putText(frame, "No face", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cam = (int(round(pan)), int(round(tilt)))
    if tracking and cam != last_sent and now - last_send_time >= SEND_INTERVAL:
        send("CAM:%d,%d" % cam)
        last_sent = cam
        last_send_time = now

    cv2.drawMarker(frame, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    cv2.putText(frame, "pan %d  tilt %d  tracking %s" % (cam[0], cam[1], "ON" if tracking else "OFF"),
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "ESP32: " + ("connected" if ser else "offline") + "   F=toggle  C=centre  Q=quit",
                (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Face follow test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("f"):
        tracking = not tracking
    if key == ord("c"):
        pan, tilt = float(PAN_CENTER), float(TILT_CENTER)
        send("CAM:%d,%d" % (PAN_CENTER, TILT_CENTER))
        last_sent = (PAN_CENTER, TILT_CENTER)

send("CAM:%d,%d" % (PAN_CENTER, TILT_CENTER))
if ser is not None:
    ser.close()
cap.release()
cv2.destroyAllWindows()
