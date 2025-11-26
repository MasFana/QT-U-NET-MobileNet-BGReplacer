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
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QGridLayout
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
        
        # Settings Utama
        self.threshold = 0.5
        self.mode = "Ganti Background"
        self.bg_image = None
        self.bg_cached = None
        self.bg_green = None
        
        # Settings Preprocessing
        self.denoise_type = "None" # None, Median, Bilateral
        self.use_clahe = False
        self.clahe_clip = 2.0
        self.show_preprocess_view = False # Toggle View
        
        # Init Objects
        self.engine = TFLiteEngine("model_quantized.tflite")
        self.clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def apply_clahe(self, img):
        """Menerapkan CLAHE pada channel L (Lightness)."""
        try:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe_obj.apply(l)
            lab_updated = cv2.merge((l_clahe, a, b))
            return cv2.cvtColor(lab_updated, cv2.COLOR_LAB2BGR)
        except Exception:
            return img

    def run(self):
        print(">> Starting Camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print(">> ERROR: Camera not found!")
            return

        # Setup Camera
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 

        prev_time = 0

        while self._run_flag:
            ret, frame = cap.read()
            if not ret: break

            h, w = frame.shape[:2]

            # ==========================================
            # 1. PIPELINE PRE-PROCESSING
            # ==========================================
            # Kita proses salinan frame untuk kebutuhan AI/Preview
            img_proc = frame.copy()

            # A. DENOISE (Bersihkan Noise Dulu)
            if self.denoise_type == "Median Blur":
                img_proc = cv2.medianBlur(img_proc, 5)
            elif self.denoise_type == "Bilateral Filter":
                # d=9, sigmaColor=75, sigmaSpace=75 (Standard smoothing)
                img_proc = cv2.bilateralFilter(img_proc, 9, 75, 75)

            # B. ENHANCE (CLAHE)
            if self.use_clahe:
                img_proc = self.apply_clahe(img_proc)

            # ==========================================
            # 2. AI INFERENCE
            # ==========================================
            # Resize & Convert processed image for Model
            img_resized = cv2.resize(img_proc, (256, 256))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_input = np.expand_dims(img_rgb.astype(np.float32), axis=0)

            # Prediksi Masker
            if self.engine.interpreter:
                pred_mask = self.engine.predict(img_input)
            else:
                pred_mask = np.zeros((256, 256, 1), dtype=np.float32)

            # ==========================================
            # 3. TAMPILAN FINAL
            # ==========================================
            final_img = None

            if self.show_preprocess_view:
                # MODE DEBUG: Tampilkan gambar hasil preprocessing
                final_img = img_proc
                
                # Info Text
                status_text = f"PREVIEW MODE | Filter: {self.denoise_type} | CLAHE: {self.use_clahe}"
                cv2.putText(final_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            else:
                # MODE NORMAL: Tampilkan hasil ganti background (Pakai Frame Asli agar tajam)
                mask_hd = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_bool = mask_hd > self.threshold

                # --- Post Processing Masker (Opsional: Bersihkan bintik masker) ---
                mask_uint8 = (mask_bool * 255).astype(np.uint8)
                kernel = np.ones((5,5), np.uint8)
                # Closing menutup lubang kecil di dalam objek
                mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
                mask_bool_clean = mask_cleaned > 127

                if self.mode == "Masker BW":
                    mask_viz = (mask_bool_clean * 255).astype(np.uint8)
                    final_img = cv2.cvtColor(mask_viz, cv2.COLOR_GRAY2BGR)

                elif self.mode == "Overlay Merah":
                    bg_mask = ~mask_bool_clean
                    red_overlay = np.zeros_like(frame)
                    red_overlay[:] = (0, 0, 255)
                    final_img = frame.copy()
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

                    # Gabungkan Frame Asli dengan Masker
                    final_img = np.where(mask_bool_clean[..., None], frame, bg_siap)

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            self.update_fps_signal.emit(fps)
            self.change_pixmap_signal.emit(final_img)

        cap.release()
        print(">> Camera Released")

    def update_settings(self, params):
        # Unpack dictionary params
        self.threshold = params['threshold']
        self.mode = params['mode']
        
        self.denoise_type = params['denoise_type']
        self.use_clahe = params['use_clahe']
        self.show_preprocess_view = params['show_preprocess']
        
        # Update CLAHE object only if needed
        new_clip = params['clahe_clip']
        if self.clahe_clip != new_clip:
            self.clahe_clip = new_clip
            self.clahe_obj = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))

    def set_background(self, filepath):
        if filepath:
            img = cv2.imread(filepath)
            if img is not None:
                self.bg_image = img
                self.bg_cached = None

    def stop(self):
        self._run_flag = False
        self.wait(2000)


# ==========================================
# 3. MAIN WINDOW
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced Background Replacer")
        self.resize(1100, 800)
        # Styling Dark Mode
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QGroupBox { border: 1px solid #444; margin-top: 10px; font-weight: bold; color: #ddd; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #ccc; }
            QPushButton { background-color: #0078d7; color: white; border-radius: 4px; padding: 6px; }
            QPushButton:hover { background-color: #0099ff; }
            QComboBox, QSlider { background-color: #333; color: white; }
        """)

        central = QWidget()
        main_layout = QVBoxLayout()
        central.setLayout(main_layout)
        self.setCentralWidget(central)

        # --- VIDEO DISPLAY ---
        self.image_label = QLabel("Initializing Camera...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #000; border: 2px solid #555;")
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        main_layout.addWidget(self.image_label, stretch=1)

        # --- PANEL KONTROL BAWAH ---
        control_panel = QWidget()
        panel_layout = QHBoxLayout()
        panel_layout.setContentsMargins(0, 0, 0, 0)
        control_panel.setLayout(panel_layout)
        main_layout.addWidget(control_panel)

        # ==========================
        # GROUP 1: PRE-PROCESSING
        # ==========================
        group_preprocess = QGroupBox("Menu Pre-Processing")
        layout_pp = QGridLayout()
        
        # 1. Filter Noise (Denoise)
        layout_pp.addWidget(QLabel("Filter Noise:"), 0, 0)
        self.combo_denoise = QComboBox()
        self.combo_denoise.addItems(["None", "Median Blur", "Bilateral Filter"])
        self.combo_denoise.setToolTip("Median: Bintik pasir. Bilateral: Halus tapi tepi tajam (Berat).")
        self.combo_denoise.currentTextChanged.connect(self.send_params)
        layout_pp.addWidget(self.combo_denoise, 0, 1)

        # 2. CLAHE (Contrast)
        self.chk_clahe = QCheckBox("Enable CLAHE")
        self.chk_clahe.stateChanged.connect(self.send_params)
        layout_pp.addWidget(self.chk_clahe, 1, 0)
        
        self.slider_clahe = QSlider(Qt.Orientation.Horizontal)
        self.slider_clahe.setRange(10, 80) # 1.0 - 8.0
        self.slider_clahe.setValue(20)
        self.slider_clahe.setToolTip("Kekuatan Kontras")
        self.slider_clahe.valueChanged.connect(self.send_params)
        layout_pp.addWidget(self.slider_clahe, 1, 1)

        # 3. Toggle View
        self.chk_show_pp = QCheckBox("Lihat Hasil Preprocess")
        self.chk_show_pp.setStyleSheet("color: #ffeb3b; font-weight: bold;")
        self.chk_show_pp.setToolTip("Tampilkan apa yang dilihat oleh AI (Blur + High Contrast)")
        self.chk_show_pp.stateChanged.connect(self.send_params)
        layout_pp.addWidget(self.chk_show_pp, 2, 0, 1, 2)

        group_preprocess.setLayout(layout_pp)
        panel_layout.addWidget(group_preprocess)

        # ==========================
        # GROUP 2: MAIN SETTINGS
        # ==========================
        group_main = QGroupBox("Pengaturan Utama")
        layout_main = QGridLayout()

        # FPS Label
        self.lbl_fps = QLabel("FPS: 0.0")
        self.lbl_fps.setStyleSheet("color: #00ff00; font-weight: bold; font-size: 14px;")
        layout_main.addWidget(self.lbl_fps, 0, 0)

        # Threshold Slider
        layout_main.addWidget(QLabel("Threshold AI:"), 1, 0)
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(1, 99)
        self.slider_thresh.setValue(50)
        self.slider_thresh.valueChanged.connect(self.send_params)
        layout_main.addWidget(self.slider_thresh, 1, 1)

        # Output Mode
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Ganti Background", "Overlay Merah", "Masker BW"])
        self.combo_mode.currentTextChanged.connect(self.send_params)
        layout_main.addWidget(self.combo_mode, 2, 0, 1, 2)

        # Load BG Button
        btn_bg = QPushButton("Pilih Background")
        btn_bg.clicked.connect(self.select_bg)
        layout_main.addWidget(btn_bg, 3, 0, 1, 2)

        group_main.setLayout(layout_main)
        panel_layout.addWidget(group_main)

        # START THREAD
        self.thread = VideoThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_fps_signal.connect(self.update_fps)
        self.thread.start()

    def update_image(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

        if self.image_label.width() > 0:
            pixmap = QPixmap.fromImage(qimg).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(pixmap)

    def update_fps(self, fps):
        self.lbl_fps.setText(f"FPS: {fps:.1f}")

    def send_params(self):
        """Mengirim semua parameter UI ke Thread sekaligus"""
        params = {
            'threshold': self.slider_thresh.value() / 100.0,
            'mode': self.combo_mode.currentText(),
            'denoise_type': self.combo_denoise.currentText(),
            'use_clahe': self.chk_clahe.isChecked(),
            'clahe_clip': self.slider_clahe.value() / 10.0,
            'show_preprocess': self.chk_show_pp.isChecked()
        }
        self.thread.update_settings(params)

    def select_bg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.thread.set_background(path)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())