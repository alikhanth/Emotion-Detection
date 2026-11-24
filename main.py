import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

COLORS = {
    'angry':    (0,   0,   220),
    'disgust':  (0,   160,   0),
    'fear':     (180,   0, 180),
    'happy':    (0,   210, 210),
    'neutral':  (180, 180, 180),
    'sad':      (220,  80,   0),
    'surprise': (0,   165, 255),
}

TEMPERATURE   = 0.6
SMOOTH_ALPHA  = 0.45
FPS_WINDOW    = 30
MIN_FACE_SIZE = 60


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int = 7) -> nn.Module:
    model = models.mobilenet_v2(weights=None)
    in_f = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_f, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes),
    )
    return model


_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(path: str = "emotion_model.pth"):
    device = get_device()
    model  = build_model()
    for p in [path, Path(__file__).parent / path]:
        if Path(p).exists():
            model.load_state_dict(
                torch.load(str(p), map_location=device, weights_only=True)
            )
            print(f"Модель загружена: {p}")
            break
    else:
        print("Файл модели не найден — случайные веса")
    return model.to(device).eval(), device


def preprocess_face(roi: np.ndarray) -> np.ndarray:
    roi = cv2.resize(roi, (128, 128))
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    lab = cv2.merge([clahe.apply(l), a, b])
    roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(roi, (0, 0), 3)
    roi = cv2.addWeighted(roi, 1.4, blurred, -0.4, 0)
    return np.clip(roi, 0, 255).astype(np.uint8)


@torch.no_grad()
def predict(model, device, face_bgr: np.ndarray) -> list:
    roi = preprocess_face(face_bgr)
    t1  = _transform(roi).unsqueeze(0).to(device)
    t2  = _transform(cv2.flip(roi, 1)).unsqueeze(0).to(device)
    avg = (model(t1) + model(t2)) / 2.0
    probs = torch.softmax(avg / TEMPERATURE, dim=1).squeeze().cpu().numpy()
    return probs.tolist()


class Smoother:
    def __init__(self, alpha: float = SMOOTH_ALPHA):
        self.alpha = alpha
        self.state = None

    def update(self, probs: list) -> np.ndarray:
        p = np.array(probs)
        if self.state is None:
            self.state = p
        else:
            self.state = self.alpha * p + (1 - self.alpha) * self.state
        return self.state / self.state.sum()

    def reset(self):
        self.state = None


def draw_face_box(frame, x, y, w, h, emotion, conf):
    color  = COLORS[emotion]
    corner = 20
    thick  = 2

    for (px, py, dx, dy) in [
        (x,   y,   1,  1),
        (x+w, y,  -1,  1),
        (x,   y+h, 1, -1),
        (x+w, y+h,-1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx*corner, py), color, thick)
        cv2.line(frame, (px, py), (px, py + dy*corner), color, thick)

    label = f"{emotion.upper()}  {conf*100:.0f}%"
    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(frame, (x, y - lh - 14), (x + lw + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)


def draw_sidebar(frame, emotion, probs, fps_buf, scale_factor):
    fh, fw = frame.shape[:2]
    sw = 265
    sx = fw - sw

    overlay = frame.copy()
    cv2.rectangle(overlay, (sx, 0), (fw, fh), (14, 14, 22), -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)
    cv2.line(frame, (sx, 0), (sx, fh), (55, 55, 75), 1)

    cv2.putText(frame, "EMOTION DETECTOR", (sx + 12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 100, 160), 1)
    cv2.putText(frame, f"sensitivity: {scale_factor:.2f}",
                (sx + 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (70, 70, 100), 1)
    cv2.line(frame, (sx + 10, 62), (fw - 10, 62), (40, 40, 60), 1)

    if emotion and probs:
        color = COLORS[emotion]
        conf  = max(probs)

        cv2.putText(frame, emotion.upper(), (sx + 14, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"{conf*100:.1f}% confidence",
                    (sx + 14, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (190, 190, 190), 1)

        cx, cy, r = fw - 38, 93, 26
        cv2.circle(frame, (cx, cy), r, (38, 38, 52), 3)
        cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, int(360 * conf), color, 3)
        cv2.putText(frame, f"{conf*100:.0f}%", (cx - 16, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1)

        cv2.line(frame, (sx + 10, 134), (fw - 10, 134), (40, 40, 60), 1)

        bar_max = sw - 92
        for i, (emo, prob) in enumerate(zip(EMOTIONS, probs)):
            by     = 146 + i * 33
            is_top = emo == emotion
            lc     = (255, 255, 255) if is_top else (95, 95, 115)
            bc     = COLORS[emo] if is_top else tuple(int(v * 0.3) for v in COLORS[emo])

            cv2.putText(frame, emo[:3].upper(), (sx + 10, by + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, lc, 1)
            cv2.rectangle(frame, (sx + 46, by),
                          (sx + 46 + bar_max, by + 14), (32, 32, 45), -1)
            filled = int(prob * bar_max)
            if filled > 0:
                cv2.rectangle(frame, (sx + 46, by),
                              (sx + 46 + filled, by + 14), bc, -1)
            if is_top and filled > 0:
                cv2.rectangle(frame, (sx + 46, by),
                              (sx + 46 + filled, by + 14), color, 1)
            cv2.putText(frame, f"{prob*100:.0f}%",
                        (sx + 46 + bar_max + 4, by + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, lc, 1)
    else:
        cv2.putText(frame, "Лицо не найдено", (sx + 14, 96),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (55, 55, 80), 1)
        cv2.putText(frame, "Встаньте перед камерой", (sx + 14, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.31, (45, 45, 65), 1)

    sep_y = fh - 88
    cv2.line(frame, (sx + 10, sep_y), (fw - 10, sep_y), (40, 40, 60), 1)
    cv2.putText(frame, "FPS", (sx + 12, sep_y + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.31, (60, 60, 90), 1)

    if len(fps_buf) >= 2:
        gx, gy = sx + 10, sep_y + 20
        gw, gh = sw - 20, 36
        mx = max(max(fps_buf), 1)
        pts = [
            (gx + int(i / (len(fps_buf) - 1) * gw),
             gy + gh - int(f / mx * gh))
            for i, f in enumerate(fps_buf)
        ]
        for i in range(1, len(pts)):
            cv2.line(frame, pts[i-1], pts[i], (45, 175, 75), 1)

    fps_now = fps_buf[-1] if fps_buf else 0
    cv2.putText(frame, f"{fps_now:.1f} fps",
                (sx + 12, fh - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (65, 195, 85), 1)
    cv2.putText(frame, "Q quit   S save   +/- sensitivity",
                (sx + 12, fh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.26, (50, 50, 72), 1)


def main():
    model, device = load_model()
    print(f"Device: {device}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother    = Smoother()
    fps_buf     = deque(maxlen=FPS_WINDOW)
    scale_factor = 1.1
    frame_n     = 0
    last_seen   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        t0    = time.perf_counter()

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=5,
            minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE),
        )

        emotion_out = None
        probs_out   = []

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            roi = frame[y: y+h, x: x+w]
            if roi.size > 0:
                raw_probs    = predict(model, device, roi)
                smooth_probs = smoother.update(raw_probs)
                idx          = int(np.argmax(smooth_probs))
                emotion_out  = EMOTIONS[idx]
                probs_out    = smooth_probs.tolist()
                last_seen    = frame_n
                draw_face_box(frame, x, y, w, h, emotion_out,
                              float(smooth_probs[idx]))
        else:
            if frame_n - last_seen > 30:
                smoother.reset()

        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        fps_buf.append(fps)

        draw_sidebar(frame, emotion_out, probs_out, list(fps_buf), scale_factor)
        cv2.imshow("Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            path = f"screenshot_{frame_n}.jpg"
            cv2.imwrite(path, frame)
            print(f"Сохранено: {path}")
        elif key in (ord("+"), ord("=")):
            scale_factor = min(1.5, round(scale_factor + 0.05, 2))
            print(f"scaleFactor: {scale_factor:.2f}")
        elif key == ord("-"):
            scale_factor = max(1.05, round(scale_factor - 0.05, 2))
            print(f"scaleFactor: {scale_factor:.2f}")

        frame_n += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()