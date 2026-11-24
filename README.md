# Emotion Detection

A real-time facial-expression classifier built with PyTorch, OpenCV, and
MobileNetV2. The application reads frames from a webcam, detects a face with
OpenCV's Haar cascade, and displays scores for seven expression classes:
angry, disgust, fear, happy, neutral, sad, and surprise.

This is a computer-vision demo, not a way to determine a person's actual
emotional state. Predictions can be inaccurate and should not be used for
high-stakes decisions.

## Features

- Webcam-based face detection and expression classification.
- CLAHE contrast adjustment and sharpening before inference.
- Original and horizontally flipped face predictions are averaged.
- Temperature scaling and exponential moving average smoothing.
- On-screen confidence bars, FPS graph, and adjustable face-detection
  sensitivity.
- Automatic device selection: Apple MPS, CUDA, or CPU.

## Project files

| File or directory | Purpose |
| --- | --- |
| `main.py` | Runs the webcam application and loads the trained model. |
| `train.py` | Trains MobileNetV2 using the image-folder dataset. |
| `emotion_model.pth` | Saved model weights used by the webcam application. |
| `training_history.json` | Per-epoch training and validation losses and accuracies. |
| `requirements.txt` | Python dependency list. |
| `data/` | Image-folder dataset used for training and validation. |

The recorded training history contains 30 epochs. Its highest validation
accuracy is approximately **67.41%** (epoch 26). This is a result from the
stored run, not a guarantee of accuracy on other data or in live use.

## Requirements and installation

Use Python 3 and install the dependencies from the project root. PyTorch and
torchvision should be installed in versions compatible with your operating
system and accelerator.

On Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file includes packages for optional data and model-analysis
workflows in addition to the dependencies used directly by the current
training and webcam scripts.

## Run the webcam application

From the project root, run:

```powershell
python main.py
```

The application opens camera device `0` and requests a 1280 × 720 frame size.
Press:

| Key | Action |
| --- | --- |
| `Q` | Quit. |
| `S` | Save the current frame as `screenshot_<frame number>.jpg`. |
| `+` or `=` | Increase the detector's scale factor, making detection less sensitive. |
| `-` | Decrease the scale factor, making detection more sensitive. |

The model file is searched for in the current working directory and alongside
`main.py`. If it is missing, the application prints a warning and continues
with randomly initialized weights, so predictions will not be meaningful.

## Train the model

Arrange the dataset in the following structure. Each class directory must
contain images, and the class directory names must be consistent between
`train` and `test`.

```text
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
```

Start training from the project root:

```powershell
python train.py --data data --epochs 30 --batch_size 64 --lr 0.001
```

All options are optional:

| Option | Default | Description |
| --- | --- | --- |
| `--data` | `data` | Directory containing the `train` and `test` folders. |
| `--epochs` | `30` | Number of training epochs. |
| `--batch_size` | `64` | Batch size for training and validation. |
| `--lr` | `0.001` | AdamW learning rate. |

The training script uses the `test` split for validation, saves the best
validation-accuracy weights to `emotion_model.pth`, and writes epoch metrics
to `training_history.json` after training completes. If the ImageNet
MobileNetV2 weights are not already cached, torchvision may need an internet
connection to download them on the first training run.

## Notes

- Run commands from the project root so relative dataset and output paths
  resolve as expected.
- Training and inference both use grayscale images converted to three
  channels and resized to 224 × 224 pixels.
- The runtime webcam application uses OpenCV's bundled frontal-face Haar
  cascade; it does not locate or crop faces from individual dataset images
  during training.
