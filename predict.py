import tensorflow as tf
import numpy as np
import cv2
import sys

# Load the trained model
model = tf.keras.models.load_model('model/waste_classifier.h5')

# Set class names (match the directory names used in training)
class_names = ['biodegradable', 'non-biodegradable']  # class_indices assumed 0: biodegradable, 1: non-biodegradable

def preprocess_frame(frame):
    img = cv2.resize(frame, (150, 150))  # Resize to match model input
    img = img / 255.0                    # Normalize
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img

def classify_frame(frame):
    processed = preprocess_frame(frame)
    prediction = model.predict(processed)[0][0]
    class_index = int(round(prediction))  # 0 or 1
    label = class_names[class_index]
    confidence = prediction if class_index == 1 else 1 - prediction
    return label, confidence

def capture_and_classify():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("✅ Webcam is working. Press 'ESC' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        label, confidence = classify_frame(frame)
        text = f"{label} ({confidence*100:.2f}%)"

        # Display the result on the video frame
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if label == 'biodegradable' else (0, 0, 255), 2)

        cv2.imshow("Waste Classifier", frame)

        if cv2.waitKey(1) == 27:  # ESC key to break
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_classify()
