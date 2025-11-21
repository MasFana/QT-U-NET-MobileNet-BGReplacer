import os
import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# --- IMPORT ENGINE ---
try:
    from ai_edge_litert.interpreter import Interpreter

    print(">> Sukses import ai_edge_litert")
except ImportError:
    print(">> ai_edge_litert tidak ditemukan, fallback ke tflite_runtime")
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter

from PyQt6.QtWidgets import QSizePolicy


# ==========================================
# 1. KELAS INFERENCE (ENGINE AI)
# ==========================================
class TFLiteEngine:
    def __init__(self, model_filename="model_quantized.tflite", num_threads=4):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_filename)

        if not os.path.exists(model_path):
            print(f"ERROR: Model not found at {model_path}")
            self.interpreter = None
            return

        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_index = self.input_details[0]["index"]
        self.output_index = self.output_details[0]["index"]
        print(">> Model Loaded Successfully")

    def predict(self, img):
        if not self.interpreter:
            return np.zeros((256, 256, 1), dtype=np.float32)
        self.interpreter.set_tensor(self.input_index, img)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]


# ==========================================
# 2. WORKER THREAD
# ==========================================
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_fps_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.threshold = 0.5
        self.mode = "Ganti Background"
        self.bg_image = None
        self.bg_cached = None
        self.bg_green = None
        self.engine = TFLiteEngine("model_quantized.tflite")

    def run(self):
        print(">> Starting Camera...")
        # FIX: Use V4L2 backend explicitly for Linux stability
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(">> ERROR: Camera not found!")
            return

        # --- PERFORMANCE SETTINGS ---
        # Force MJPG (Motion JPEG) to unlock 30 FPS on USB 2.0 ports
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # FIX: Disable Auto Exposure (Common cause of low FPS in low light)
        # 0.25 or 0.75 usually means 'Manual' on V4L2, exact value varies by camera
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

        prev_time = 0
        print(">> Camera Loop Started")

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                print(">> Frame Read Error")
                break

            h, w = frame.shape[:2]

            # --- 1. PREPROCESSING ---
            img_resized = cv2.resize(frame, (256, 256))
            # FIX: BGR to RGB (Critical for TFLite MobileNet colors)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # Input Range: 0-255 float32 (Based on your metrics.ipynb)
            img_input = np.expand_dims(img_rgb.astype(np.float32), axis=0)

            # --- 2. INFERENCE ---
            if self.engine.interpreter:
                pred_mask = self.engine.predict(img_input)
            else:
                pred_mask = np.zeros((256, 256, 1), dtype=np.float32)

            # --- 3. POST-PROCESSING ---
            mask_hd = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_hd > self.threshold

            final_img = frame

            if self.mode == "Masker BW":
                mask_viz = (mask_bool * 255).astype(np.uint8)
                final_img = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)

            elif self.mode == "Overlay Merah":
                # Optimization: Apply redness only to background pixels
                # mask_bool = True (Person), False (Bg)
                # We want to color the False area
                bg_mask = ~mask_bool

                # Create a red overlay
                red_overlay = np.zeros_like(frame)
                red_overlay[:] = (0, 0, 255)

                # Copy frame to avoid modifying original buffer directly
                final_img = frame.copy()

                # Fast blending using boolean indexing
                # Blend only where bg_mask is True
                final_img[bg_mask] = cv2.addWeighted(
                    frame[bg_mask], 0.7, red_overlay[bg_mask], 0.3, 0
                )

            elif self.mode == "Ganti Background":
                bg_siap = None
                if self.bg_image is not None:
                    if self.bg_cached is None or self.bg_cached.shape[:2] != (h, w):
                        self.bg_cached = cv2.resize(self.bg_image, (w, h))
                    bg_siap = self.bg_cached
                else:
                    if self.bg_green is None or self.bg_green.shape[:2] != (h, w):
                        self.bg_green = np.zeros_like(frame)
                        self.bg_green[:] = (0, 255, 0)
                    bg_siap = self.bg_green

                # np.where is fast!
                final_img = np.where(mask_bool[..., None], frame, bg_siap)

            # --- 4. FPS CALC ---
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            self.update_fps_signal.emit(fps)
            self.change_pixmap_signal.emit(final_img)

        # CLEANUP
        cap.release()
        print(">> Camera Released")

    def update_settings(self, threshold, mode):
        self.threshold = threshold
        self.mode = mode

    def set_background(self, filepath):
        if filepath:
            img = cv2.imread(filepath)
            if img is not None:
                self.bg_image = img
                self.bg_cached = None
                print(">> Background Set")

    def stop(self):
        self._run_flag = False
        # We do NOT call self.wait() here directly if called from UI thread
        # because it might freeze GUI if camera is stuck.
        # But for clean exit, we usually do.
        self.wait(2000)  # Wait max 2 seconds


# ==========================================
# 3. MAIN WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("U-NET MobileNet Background Replacer")
        self.resize(1000, 750)
        self.setStyleSheet(
            "QMainWindow { background-color: #222; color: white; } QLabel { color: white; }"
        )

        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.image_label = QLabel("Initializing...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "background-color: black; border: 2px solid #555;"
        )

        # --- FIX: PREVENT WINDOW GROWTH LOOP ---
        # Tell the layout: "Don't resize the window based on this label's content."
        self.image_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )

        layout.addWidget(
            self.image_label, stretch=1
        )  # Add stretch to take available space

        controls = QHBoxLayout()
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet("font-weight: bold; color: #0f0; font-size: 16px;")
        controls.addWidget(self.lbl_fps)

        controls.addWidget(QLabel("Threshold:"))
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(1, 99)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_params)
        controls.addWidget(self.slider)

        self.combo = QComboBox()
        self.combo.addItems(["Ganti Background", "Overlay Merah", "Masker BW"])
        self.combo.currentTextChanged.connect(self.update_params)
        controls.addWidget(self.combo)

        btn = QPushButton("Load BG")
        btn.setStyleSheet("background-color: #007bff; color: white; padding: 5px;")
        btn.clicked.connect(self.select_bg)
        controls.addWidget(btn)

        layout.addLayout(controls)

        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_fps_signal.connect(self.update_fps)
        self.thread.start()

    def update_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        # .copy() prevents memory leaks
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

        # Check if label has size (it might be 0 during init)
        if self.image_label.width() > 0 and self.image_label.height() > 0:
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(pixmap)
        else:
            self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def update_fps(self, fps):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def update_params(self):
        self.thread.update_settings(
            self.slider.value() / 100.0, self.combo.currentText()
        )

    def select_bg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select BG", "", "Images (*.png *.jpg)"
        )
        if path:
            self.thread.set_background(path)

    def closeEvent(self, event):
        print(">> Shutting down...")
        self.thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
