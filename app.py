import cv2
import sys
import numpy as np
import threading
import queue
import easyocr
import time
import textwrap
import json
import os
import fitz  # PyMuPDF
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QDialog, QFormLayout, QLineEdit, QCheckBox, QDialogButtonBox, QMessageBox, QFileDialog, QLabel
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap
from PyQt6.QtCore import Qt

is_running = True
main_config = {}

def find_capture_device():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened(): return cap
    return None

def load_corrections(filepath):
    correction_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) == 2:
                        error, correction = parts
                        correction_dict[error.strip()] = correction.strip()
    except FileNotFoundError:
        pass
    return correction_dict

def post_process_text(ocr_result, wrapper, correction_dict):
    paragraphs = []
    current_paragraph = ""
    for item in ocr_result:
        line = item[1].strip()
        current_paragraph += line + " "
        if line.endswith(('.', '?', '!', ':')):
            paragraphs.append(current_paragraph.strip()); current_paragraph = ""
    if current_paragraph: paragraphs.append(current_paragraph.strip())
    corrected_paragraphs = []
    for p in paragraphs:
        temp_p = p
        for error, correction in correction_dict.items():
            temp_p = temp_p.replace(error, correction)
        corrected_paragraphs.append(temp_p)
    wrapper.width = main_config.get('text_wrap_width', 70)
    wrapped_paragraphs = [wrapper.fill(p) for p in corrected_paragraphs]
    return "\n\n".join(wrapped_paragraphs)

def ocr_worker(job_q, result_q, reader, wrapper, correction_dict):
    while True:
        try:
            frame = job_q.get(timeout=1)
        except queue.Empty:
            if not is_running: break
            continue
        try:
            if frame is None: break
            result = reader.readtext(frame)
            formatted_text = post_process_text(result, wrapper, correction_dict) if result else "[No text found]"
            result_q.put(formatted_text)
        except Exception as e:
            result_q.put(f"[OCR Error: {e}]")
        finally:
            job_q.task_done()

class ModeSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("모드 선택")
        self.layout = QVBoxLayout(self)
        self.selected_mode = None
        self.btn_ocr = QPushButton("1. 실시간 OCR (Live OCR)")
        self.btn_image = QPushButton("2. 이미지 저장 (Image Capture)")
        self.btn_pdf = QPushButton("3. PDF 파일 처리 (PDF Batch Process)")
        self.btn_ocr.clicked.connect(lambda: self.set_mode('ocr'))
        self.btn_image.clicked.connect(lambda: self.set_mode('image'))
        self.btn_pdf.clicked.connect(lambda: self.set_mode('pdf'))
        self.layout.addWidget(QLabel("실행할 작업 모드를 선택하세요:"))
        self.layout.addWidget(self.btn_ocr); self.layout.addWidget(self.btn_image); self.layout.addWidget(self.btn_pdf)

    def set_mode(self, mode):
        self.selected_mode = mode
        self.accept()

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.config = config.copy()
        self.layout = QFormLayout(self)
        self.widgets = {}
        descriptions = {
            'ocr_languages': "OCR 언어 (쉼표로 구분) *재시작 필요", 'gpu_enabled': "GPU 가속 사용 여부 *재시작 필요",
            'motion_threshold': "움직임 감지 기준 값", 
            'stability_threshold_frames': "안정화로 판단하기 위한 프레임 수", 'user_cooldown_seconds': "캡처 후 다음 감지까지의 대기 시간 (초)",
            'text_wrap_width': "결과 텍스트의 한 줄 최대 글자 수"
        }
        for key, value in self.config.items():
            frame = QWidget(); row_layout = QVBoxLayout(frame); row_layout.setContentsMargins(0,0,0,5)
            if isinstance(value, bool): widget = QCheckBox(key); widget.setChecked(value)
            else:
                widget = QLineEdit(); widget.setPlaceholderText(key)
                widget.setText(",".join(value) if isinstance(value, list) else str(value))
            desc_label = QPushButton(descriptions.get(key, ""), enabled=False); desc_label.setStyleSheet("Text-align:left; border:0; color:grey;")
            row_layout.addWidget(widget); row_layout.addWidget(desc_label)
            self.widgets[key] = widget; self.layout.addRow(frame)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.on_save); self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

    def on_save(self):
        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox): self.config[key] = widget.isChecked()
            elif key == 'ocr_languages': self.config[key] = [lang.strip() for lang in widget.text().split(',')]
            else:
                try: self.config[key] = int(widget.text())
                except ValueError: self.config[key] = float(widget.text())
        with open("config.json", "w", encoding="utf-8") as f: json.dump(self.config, f, indent=4)
        QMessageBox.information(self, "저장 완료", "설정이 저장되었습니다. 일부 설정은 프로그램을 재시작해야 적용됩니다.")
        super().accept()

class MainWindow(QMainWindow):
    def __init__(self, config, mode):
        super().__init__()
        self.config = config; self.mode = mode
        self.pdf_saved_this_session = False
        self.setWindowTitle("OCR Application")
        self.setGeometry(100, 100, 1200, 700)
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.video_label = QLabel("카메라 로딩 중..."); self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.video_label.setMinimumWidth(640)
        main_layout.addWidget(self.video_label, 2)
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget); main_layout.addWidget(right_widget, 1)
        self.ocr_results_area = QTextEdit(); self.ocr_results_area.setReadOnly(True); self.ocr_results_area.setFontPointSize(14)
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True); self.log_area.setMaximumHeight(100)
        right_layout.addWidget(QLabel("OCR 결과 / 저장된 파일 목록:")); right_layout.addWidget(self.ocr_results_area)
        right_layout.addWidget(QLabel("로그:")); right_layout.addWidget(self.log_area)
        button_layout = QHBoxLayout()
        self.copy_button = QPushButton("클립보드로 복사"); self.clear_button = QPushButton("내용/목록 지우기")
        self.save_as_button = QPushButton("텍스트 파일로 저장"); self.pdf_button = QPushButton("PDF로 저장")
        self.settings_button = QPushButton("설정")
        button_layout.addWidget(self.copy_button); button_layout.addWidget(self.clear_button); button_layout.addWidget(self.save_as_button); button_layout.addWidget(self.pdf_button)
        button_layout.addStretch(); button_layout.addWidget(self.settings_button)
        right_layout.addLayout(button_layout)
        self.copy_button.clicked.connect(self.copy_to_clipboard); self.clear_button.clicked.connect(self.clear_text)
        self.save_as_button.clicked.connect(self.save_as); self.pdf_button.clicked.connect(self.compile_to_pdf)
        self.settings_button.clicked.connect(self.open_settings)
        self.update_ui_for_mode()

    def update_ui_for_mode(self):
        if self.mode == 'ocr': self.pdf_button.setEnabled(False)
        else: self.copy_button.setEnabled(False); self.save_as_button.setEnabled(False)

    def update_video_frame(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def add_ocr_text(self, text): self.ocr_results_area.append(text); self.ocr_results_area.verticalScrollBar().setValue(self.ocr_results_area.verticalScrollBar().maximum())
    def add_log(self, text): self.log_area.append(text); self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())
    def copy_to_clipboard(self): QGuiApplication.clipboard().setText(self.ocr_results_area.toPlainText()); self.add_log("UI 창의 내용이 클립보드에 복사되었습니다.")
    def clear_text(self): self.ocr_results_area.clear()
    def save_as(self):
        text_to_save = self.ocr_results_area.toPlainText()
        if not text_to_save: QMessageBox.warning(self, "경고", "저장할 내용이 없습니다."); return
        path, _ = QFileDialog.getSaveFileName(self, "텍스트 파일로 저장", "output.txt", "Text Files (*.txt)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f: f.write(text_to_save)
                self.add_log(f"파일이 성공적으로 저장되었습니다: {path}")
            except Exception as e: self.add_log(f"파일 저장 중 오류가 발생했습니다: {e}")

    def open_settings(self):
        global main_config
        dialog = SettingsDialog(main_config, self)
        if dialog.exec():
            main_config = dialog.config
            self.add_log("설정 변경이 감지되었습니다. 적용을 위해 프로그램을 재시작하세요.")

    def compile_to_pdf(self):
        self.add_log("PDF 생성을 시작합니다...")
        image_folder = 'captures'
        if not os.path.exists(image_folder): self.add_log(f"경고: '{image_folder}' 폴더를 찾을 수 없습니다."); return
        image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.png')])
        if not image_files: QMessageBox.warning(self, "경고", "PDF로 만들 이미지가 captures 폴더에 없습니다."); return
        path, _ = QFileDialog.getSaveFileName(self, "PDF로 저장", "output.pdf", "PDF Files (*.pdf)")
        if path:
            try:
                pil_images = [Image.open(p).convert('RGB') for p in image_files]
                if pil_images:
                    pil_images[0].save(path, save_all=True, append_images=pil_images[1:])
                    self.add_log(f"PDF 파일이 성공적으로 저장되었습니다: {path}")
                    self.pdf_saved_this_session = True
            except Exception as e: self.add_log(f"PDF 생성 중 오류가 발생했습니다: {e}")

    def closeEvent(self, event):
        global is_running
        is_running = False
        event.accept()

def process_pdf_mode(config, app):
    path, _ = QFileDialog.getOpenFileName(None, "처리할 PDF 파일 선택", "", "PDF Files (*.pdf)")
    if not path: print("PDF 파일이 선택되지 않았습니다. 프로그램을 종료합니다."); return
    main_window = QMainWindow(); main_window.setWindowTitle("PDF 처리 중..."); text_area = QTextEdit("PDF 처리 중... 잠시만 기다려주세요."); main_window.setCentralWidget(text_area); main_window.show()
    app.processEvents()
    reader = easyocr.Reader(config['ocr_languages'], gpu=config['gpu_enabled'])
    correction_dict = load_corrections("corrections.txt")
    wrapper = textwrap.TextWrapper(width=config['text_wrap_width'], break_long_words=False, replace_whitespace=False)
    try:
        doc = fitz.open(path)
        full_text = ""
        for i, page in enumerate(doc):
            text_area.setText(f"총 {len(doc)} 페이지 중 {i+1} 페이지 처리 중..."); app.processEvents()
            pix = page.get_pixmap(dpi=300)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            if pix.n == 4: img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
            result = reader.readtext(img_data)
            if result: full_text += post_process_text(result, wrapper, correction_dict) + "\n\n"
        output_filename = os.path.splitext(os.path.basename(path))[0] + "_output.txt"
        with open(output_filename, "w", encoding="utf-8") as f: f.write(full_text)
        QMessageBox.information(None, "처리 완료", f"PDF 처리가 완료되었습니다!\n결과가 '{output_filename}' 파일에 저장되었습니다.")
    except Exception as e: QMessageBox.critical(None, "오류", f"PDF 처리 중 오류가 발생했습니다: {e}")
    finally: main_window.close()

def live_capture_mode(config, mode, app):
    global is_running, main_config
    main_config = config
    main_window = MainWindow(config, mode); main_window.show()
    main_window.add_log("실시간 캡처 모드로 시작합니다.")
    cap = find_capture_device()
    if cap is None: main_window.add_log("오류: 캡처 장치를 찾을 수 없습니다."); return
    main_window.add_log(f"캡처 장치 로드 완료.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, first_frame = cap.read()
    if not ret: main_window.add_log("오류: 카메라 프레임을 읽을 수 없습니다."); return
    roi_text_img = first_frame.copy(); cv2.putText(roi_text_img, "Draw ROI and Press ENTER", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    roi = cv2.selectROI("ROI 선택", roi_text_img, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI 선택")
    roi_x, roi_y, roi_w, roi_h = [int(c) for c in roi]
    if roi_w == 0 or roi_h == 0: main_window.add_log("경고: ROI가 선택되지 않았습니다. 전체 화면으로 진행합니다."); roi_x, roi_y, roi_w, roi_h = 0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    main_window.add_log(f"관심 영역 선택 완료: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
    
    job_queue, result_queue, ocr_thread, saved_image_paths = None, None, None, []
    if mode == 'ocr':
        main_window.add_log("EasyOCR 모델을 로드하는 중입니다..."); app.processEvents()
        reader = easyocr.Reader(config['ocr_languages'], gpu=config['gpu_enabled'])
        main_window.add_log(f"EasyOCR 모델 로드 완료. [실행 장치: {reader.device}]")
        correction_dict = load_corrections("corrections.txt"); main_window.add_log(f"{len(correction_dict)}개의 교정 규칙을 로드했습니다.")
        wrapper = textwrap.TextWrapper(width=config['text_wrap_width'], break_long_words=False, replace_whitespace=False)
        job_queue = queue.Queue(); result_queue = queue.Queue()
        ocr_thread = threading.Thread(target=ocr_worker, args=(job_queue, result_queue, reader, wrapper, correction_dict), daemon=True); ocr_thread.start()
    else: # Image mode
        if not os.path.exists('captures'): os.makedirs('captures')

    status, is_flipping, mean_diff, last_capture_time = "Ready", False, 0.0, 0
    page_counter, previous_frame_gray, stability_counter = 0, None, 0
    STATUS_COLORS = { "Ready": (0, 255, 0), "Flipping...": (0, 255, 255), "Stabilizing...": (255, 255, 0), "OCR Queued": (255, 0, 0), "Saved!": (255, 0, 255), "Image Saved!": (0, 165, 255) }
    
    output_file = None
    try:
        if mode == 'ocr':
            output_file = open("output.txt", "a", encoding="utf-8")

        while is_running:
            ret, frame = cap.read();
            if not ret: break
            display_frame = frame.copy()
            clean_roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
            cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            if mode == 'ocr':
                try:
                    ocr_text = result_queue.get_nowait()
                    page_counter += 1
                    ui_output = f"--- Page {page_counter} ---\n{ocr_text}"
                    main_window.add_ocr_text(ui_output)
                    if output_file: output_file.write(ocr_text + "\n\n"); output_file.flush()
                    main_window.add_log(f"Page {page_counter} 처리 및 저장 완료.")
                    last_capture_time = time.time(); status = "Saved!"
                    result_queue.task_done()
                except queue.Empty:
                    pass
            gray = cv2.cvtColor(clean_roi_frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if previous_frame_gray is not None:
                diff = cv2.absdiff(previous_frame_gray, gray); mean_diff = np.mean(diff)
                if mean_diff > config['motion_threshold'] and not is_flipping and (time.time() - last_capture_time > config['user_cooldown_seconds']):
                    status = "Flipping..."; is_flipping = True
                if is_flipping:
                    if mean_diff <= config['motion_threshold']:
                        stability_counter += 1
                        status = "Stabilizing..."
                    else:
                        stability_counter = 0; status = "Flipping..."
                    if stability_counter >= config['stability_threshold_frames']:
                        if mode == 'ocr':
                            job_queue.put(clean_roi_frame.copy()); status = "OCR Queued"
                        else:
                            page_counter += 1
                            filename = f"captures/capture_{page_counter:04d}.png"
                            cv2.imwrite(filename, clean_roi_frame)
                            saved_image_paths.append(filename)
                            main_window.add_ocr_text(f"{filename}")
                            main_window.add_log(f"이미지 저장됨: {filename}")
                            status = "Image Saved!"; last_capture_time = time.time()
                        is_flipping = False; stability_counter = 0
                current_status_key = status.split('!')[0] + '!' if '!' in status else status
                if not is_flipping and current_status_key not in ["Saved!", "OCR Queued", "Image Saved!"]: status = "Ready"
                elif status in ["Saved!", "Image Saved!"] and (time.time() - last_capture_time > config['user_cooldown_seconds']):
                    status = "Ready"
            status_color = STATUS_COLORS.get(status, (0,0,255))
            cv2.putText(display_frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Difference: {mean_diff:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Stability: {stability_counter}/{config['stability_threshold_frames']}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            main_window.update_video_frame(display_frame)
            previous_frame_gray = gray.copy()
            app.processEvents()
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: is_running = False
    finally:
        main_window.add_log("프로그램 종료 중...")
        if output_file: output_file.close()
        if mode == 'ocr' and job_queue is not None:
            job_queue.put(None)
            ocr_thread.join(timeout=5)
        elif mode == 'image' and saved_image_paths:
            if not main_window.pdf_saved_this_session:
                if QMessageBox.question(main_window, "PDF 생성", f"{len(saved_image_paths)}개의 이미지를 PDF로 변환하시겠습니까?") == QMessageBox.StandardButton.Yes:
                    main_window.compile_to_pdf()
        cap.release()
        if 'app' in locals(): app.quit()

def main():
    global is_running, main_config
    app = QApplication(sys.argv)
    mode_dialog = ModeSelectionDialog()
    if not mode_dialog.exec(): sys.exit(0)
    mode_choice = mode_dialog.selected_mode
    try:
        with open("config.json", "r", encoding="utf-8") as f: main_config = json.load(f)
    except FileNotFoundError:
        main_config = { "ocr_languages": ["ko", "en"], "gpu_enabled": True, "motion_threshold": 1.5, "stabilization_delay_seconds": 0.5, "stability_threshold_frames": 5, "user_cooldown_seconds": 0.4, "text_wrap_width": 70 }

    if mode_choice == 'pdf':
        process_pdf_mode(main_config, app)
        sys.exit(0)
    else:
        live_capture_mode(main_config, mode_choice, app)

if __name__ == "__main__":
    main()
