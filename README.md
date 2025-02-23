# Project Documentation

Welcome to the project! Below are links to the implementation documentation for the backend, machine learning models and frontend:

- 📌 **Backend Documentation**: [Go here](backend/README.md)
- 📌 **Machine Learning Models Documentation**: [Go here](backend/machine_learning_models/Readme.md)
- 🎨 **Frontend Documentation**: [Go here](frontend/README.md)

Each section contains detailed information about dependencies, and usage.

Happy coding! 🚀

# Generate Music from Humming

This documentation provides an overview of the "Generate Music from Humming" application. These images depict different stages of music generation, pitch detection, and model metrics.

---

## Filtering Detected Pitches
Filtering is the first step in the pitch detection pipeline, where unwanted frequencies are removed to enhance accuracy.

![Filtering Detected Pitches](frontend/src/assets/Pitch_detection.png)


---

## Original Annotations
This image shows the original pitch annotations, which serve as a ground truth for evaluating the pitch detection model.

![Original Annotations](frontend/src/assets/Original_annotations.png)


---

## Comparing Original and Detected Pitches
This comparison helps in analyzing the accuracy of the detected pitches against the original annotations.

![Comparing Original and Detected Pitches](frontend/src/assets/Compare_detected_vs_Original.png)


---

## Smoothened Pitches
After detecting pitches, they are smoothened to improve the overall melody flow.

![Smoothened Pitches](frontend/src/assets/Predicted_annotations_after_smoothing.png)


---

## Polyphonic Music
This plot represents the polyphonic generated music, showcasing multiple notes played simultaneously.

![Polyphonic Music](frontend/src/assets/Polyphonic_Midi_music.png)


---

## CLSTM Predictions
The Conditional LSTM (CLSTM) model's predictions for the music sequence are displayed here.

![CLSTM Predictions](frontend/src/assets/Predictions_for_CLSTM.png)


---

## CVAE Metrics

![CVAE Metrics](frontend/src/assets/Metrics_for_CVAE.png)


---

## CLSTM Metrics

![CLSTM Metrics](frontend/src/assets/Metrics_for_CLSTM.png)



