# Hand Gesture Recognition

Real-time hand gesture recognition using a CNN trained with TensorFlow 
and OpenCV for live webcam inference.

## Pipeline
1. `fetch_images.py` — fetch base images
2. `collect_data.py` — collect custom gesture data via webcam
3. `prepare_data.py` — preprocess and label data
4. `create_model.py` — train CNN with TensorFlow
5. `live_recognition.py` — real-time recognition via webcam

## Requirements
```bash
pip install -r requirements.txt
```

## Run
```bash
python live_recognition.py
```

## Gestures Supported
- thumbs up
- peace
- flat hand
- fist

## Results
- Training accuracy: X%
- Validation accuracy: X%
