import sys
import cv2
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QRadioButton, QFileDialog, QProgressBar
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap, QColor

from fer import FER # Assuming 'fer' library is installed

# --- Emotion Detection Thread ---
class VideoWorker(QThread):
    # Signals to send data back to the GUI thread
    image_updated = pyqtSignal(QImage)
    fatigue_updated = pyqtSignal(int) # Percentage 0-100
    emotions_updated = pyqtSignal(dict) # Dictionary of emotion probabilities
    status_message = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.video_source = 0 # 0 for webcam, path for file
        self.cap = None
        self.emotion_detector = FER(mtcnn=True) # Use MTCNN for face detection
        self.fatigue_history = [] # To average fatigue over time
        self.history_length = 30 # Number of frames to average

    def set_video_source(self, source):
        self.video_source = source

    def run(self):
        self.running = True
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            self.status_message.emit(f"Error: Could not open video source {self.video_source}")
            self.running = False
            return

        self.status_message.emit("Video stream started. Detecting emotions...")

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.video_source, str): # End of video file
                    self.status_message.emit("Video file ended.")
                    self.running = False
                else: # Webcam error
                    self.status_message.emit("Error: Lost webcam feed.")
                    self.running = False
                break

            # Convert to RGB for FER and PyQt
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # --- Emotion Detection ---
            # The fer library handles face detection internally if mtcnn=True
            results = self.emotion_detector.detect_emotions(rgb_frame)

            current_emotions = {}
            current_fatigue_score = 0.0

            if results:
                # Assuming one main face, take the first one
                face_data = results[0]
                emotions = face_data['emotions']
                bbox = face_data['box'] # (x, y, w, h)

                # Store current emotions for display
                current_emotions = emotions

                # Calculate fatigue score
                # Weights for fatigue: Sad, Neutral, Disgust are positive contributors
                # Happy is a negative contributor
                W_SAD = 0.4
                W_NEUTRAL = 0.2
                W_DISGUST = 0.1
                W_HAPPY = 0.3 # Subtracts from fatigue

                fatigue_raw = (W_SAD * emotions.get('sad', 0)) + \
                              (W_NEUTRAL * emotions.get('neutral', 0)) + \
                              (W_DISGUST * emotions.get('disgust', 0)) - \
                              (W_HAPPY * emotions.get('happy', 0))

                # Normalize fatigue_raw to a 0-1 range (adjust min/max as needed)
                # Example: If min possible is -0.3 and max is 0.7
                min_fatigue_raw = -0.3 # Example: 0% Sad, 0% Neutral, 0% Disgust, 100% Happy
                max_fatigue_raw = 0.7  # Example: 100% Sad, 0% Happy, etc.
                
                # Clip and normalize to 0-1
                fatigue_normalized = max(0, min(1, (fatigue_raw - min_fatigue_raw) / (max_fatigue_raw - min_fatigue_raw)))

                self.fatigue_history.append(fatigue_normalized)
                if len(self.fatigue_history) > self.history_length:
                    self.fatigue_history.pop(0)
                
                # Average over history
                averaged_fatigue = sum(self.fatigue_history) / len(self.fatigue_history)
                current_fatigue_score = int(averaged_fatigue * 100) # Convert to percentage

                # Draw bounding box and emotion on frame (for visualization)
                x, y, w, h = bbox
                cv2.rectangle(rgb_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display dominant emotion or fatigue score on face
                dominant_emotion = max(emotions, key=emotions.get)
                text = f"{dominant_emotion} ({current_fatigue_score}%)"
                cv2.putText(rgb_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                self.status_message.emit("Face detected. Processing emotions...")

            else:
                self.status_message.emit("No face detected.")
                # If no face, reset fatigue history or slowly decay
                self.fatigue_history = [] # Or decay gradually

            # Convert RGB frame to QImage and emit
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.image_updated.emit(q_img)
            self.fatigue_updated.emit(current_fatigue_score)
            self.emotions_updated.emit(current_emotions)

            # Small delay to prevent burning CPU (adjust as needed for high resolution)
            QThread.msleep(30) # ~33 FPS without heavy processing

        self.cap.release()
        self.status_message.emit("Video stream stopped.")


    def stop(self):
        self.running = False
        self.wait() # Wait for the thread to finish its current loop

# --- Main GUI Application ---
class MentalFatigueApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mental Fatigue Estimator")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.init_ui()
        self.worker_thread = VideoWorker()
        self.worker_thread.image_updated.connect(self.update_video_feed)
        self.worker_thread.fatigue_updated.connect(self.update_fatigue_bar)
        self.worker_thread.emotions_updated.connect(self.update_emotion_display)
        self.worker_thread.status_message.connect(self.update_status_bar)

    def init_ui(self):
        # Video Display
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480) # Standard video resolution
        self.video_label.setStyleSheet("background-color: black; color: white; border: 1px solid gray;")
        self.main_layout.addWidget(self.video_label)

        # Controls Layout
        controls_layout = QHBoxLayout()
        
        # Input Source Selection
        self.webcam_radio = QRadioButton("Webcam")
        self.webcam_radio.setChecked(True)
        self.video_file_radio = QRadioButton("Video File")
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.select_video_file)
        self.browse_button.setEnabled(False) # Enabled only when video file is selected

        self.webcam_radio.toggled.connect(lambda: self.browse_button.setEnabled(not self.webcam_radio.isChecked()))
        self.video_file_radio.toggled.connect(lambda: self.browse_button.setEnabled(self.video_file_radio.isChecked()))

        controls_layout.addWidget(self.webcam_radio)
        controls_layout.addWidget(self.video_file_radio)
        controls_layout.addWidget(self.browse_button)
        controls_layout.addStretch()

        # Start/Stop Button
        self.start_stop_button = QPushButton("Start Detection")
        self.start_stop_button.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.start_stop_button)

        self.main_layout.addLayout(controls_layout)

        # Fatigue Indicator
        fatigue_layout = QHBoxLayout()
        fatigue_label = QLabel("Mental Fatigue:")
        self.fatigue_progress_bar = QProgressBar()
        self.fatigue_progress_bar.setRange(0, 100)
        self.fatigue_progress_bar.setValue(0)
        self.fatigue_progress_bar.setTextVisible(True) # Show percentage

        # Custom styling for the color bar
        # Green (low fatigue) -> Yellow (medium) -> Red (high)
        self.fatigue_progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                  stop:0 green, stop:0.5 yellow, stop:1 red);
            }
            """
        )
        
        fatigue_layout.addWidget(fatigue_label)
        fatigue_layout.addWidget(self.fatigue_progress_bar)
        self.main_layout.addLayout(fatigue_layout)

        # Emotion Probabilities Display
        emotion_display_layout = QHBoxLayout()
        self.emotion_labels = {}
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        for emotion in emotions:
            label = QLabel(f"{emotion}: 0%")
            self.emotion_labels[emotion] = label
            emotion_display_layout.addWidget(label)
        emotion_display_layout.addStretch() # Push labels to left
        self.main_layout.addLayout(emotion_display_layout)


        # Status Bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready to start.")

        self.video_file_path = None

    def select_video_file(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.video_file_path = selected_files[0]
                self.status_bar.showMessage(f"Selected video: {self.video_file_path}")
            else:
                self.video_file_path = None
                self.status_bar.showMessage("No video file selected.")

    def toggle_detection(self):
        if self.worker_thread.running:
            self.worker_thread.stop()
            self.start_stop_button.setText("Start Detection")
            self.status_bar.showMessage("Detection stopped.")
            self.video_label.setText("Video Feed") # Clear video display
            self.fatigue_progress_bar.setValue(0)
            for label in self.emotion_labels.values():
                label.setText("0%")

        else:
            if self.webcam_radio.isChecked():
                self.worker_thread.set_video_source(0) # Webcam
            elif self.video_file_radio.isChecked():
                if self.video_file_path:
                    self.worker_thread.set_video_source(self.video_file_path)
                else:
                    self.status_bar.showMessage("Please select a video file first.")
                    return
            
            self.worker_thread.start()
            self.start_stop_button.setText("Stop Detection")
            self.status_bar.showMessage("Starting detection...")

    def update_video_feed(self, q_image):
        pixmap = QPixmap.fromImage(q_image)
        # Scale pixmap to fit the label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)

    def update_fatigue_bar(self, fatigue_percentage):
        self.fatigue_progress_bar.setValue(fatigue_percentage)
        # Update progress bar color based on fatigue level (more sophisticated logic needed here)
        # This basic QProgressBar styling will apply a gradient.
        # For more control over distinct color segments, you'd need to create a custom widget or
        # adjust the QProgressBar::chunk style dynamically based on the value.
        
    def update_emotion_display(self, emotions):
        for emotion, score in emotions.items():
            if emotion in self.emotion_labels:
                self.emotion_labels[emotion].setText(f"{emotion}: {score:.1f}%")

    def update_status_bar(self, message):
        self.status_bar.showMessage(message)

    def closeEvent(self, event):
        if self.worker_thread.running:
            self.worker_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MentalFatigueApp()
    window.show()
    sys.exit(app.exec())
