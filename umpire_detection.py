import argparse 
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# output a frame every interval from video feed
def _sample_frames(source, interval_ms=500):
    cap = cv2.VideoCapture(source)
    is_file = Path(source).exists()
    start = time.monotonic()
    next_ms = 0
    while cap.grab():
        if is_file:
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) # find current position of video, in ms
        else:
            ts = (time.monotonic() - start) * 1000 # if its a stream, use device time to find elapsed time, in ms

        if ts < next_ms: 
            continue
        ok, frame = cap.retrieve()
        if ok:
            yield ts, frame
            next_ms = ts + interval_ms
    cap.release()

# crops a square around the box
# sliding it back into the frame if necessary
def _crop_square(frame, box, margin=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    side = int(min(max(x2 - x1, y2 - y1) * (1 + margin), h, w))
    x = int(min(max((x1 + x2) / 2 - side / 2, 0), w - side))
    y = int(min(max((y1 + y2) / 2 - side / 2, 0), h - side))
    return frame[y:y + side, x:x + side]

# yields timestamps_ms, confidence, and cropped image 
# for each frame WITH an umpire 
def _detect_and_crop(source, weights, conf=0.7, interval_ms=500):
    model = YOLO(weights)
    for ts, frame in _sample_frames(source, interval_ms):
        result = model.predict(
            frame, 
            conf=conf, 
            imgsz=640, 
            classes=[0],
            verbose=False
            )[0]
        if len(result.boxes) == 0:
            continue
        best = int(result.boxes.conf.argmax())
        box = result.boxes.xyxy[best].tolist()
        yield ts, float(result.boxes.conf[best]), _crop_square(frame, box)


def umpire_detection(
    source, 
    weights = f'datasets/tertiary/runs/2/weights/best.pt', 
    out = "../crops", 
    conf = 0.8, 
    interval = 500
    ):

    for ts, conf, crop in _detect_and_crop(source, weights, conf, interval):
        path = f'{out}/{int(ts):08d}ms_{conf:.2f}.jpg'
        cv2.imwrite(path, crop)
        print(f'umpire at {ts/1000:.1f}s (conf {conf:.2f})')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Path to video")
    p.add_argument("--weights", default=f'datasets/tertiary/runs/2/weights/best.pt')
    p.add_argument("--out", default="../crops")
    p.add_argument("--conf", type=float, default=0.8)
    p.add_argument("--interval", type=int, default=500, help="Milliseconds between frames")
    args = p.parse_args()
 
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    umpire_detection(args.source, args.weights, args.out, args.conf, args.interval)