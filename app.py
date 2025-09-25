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
from PIL import Image
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QDialog, QFormLayout, QLineEdit, QCheckBox, QDialogButtonBox, QMessageBox, QFileDialog
from PyQt6.QtGui import QGuiApplication

is_running = True
main_config = {}

def find_capture_device():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
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
        print(f"'{filepath}'에서 {len(correction_dict)}개의 교정 규칙을 로드했습니다.")
    except FileNotFoundError:
        print(f"경고: 교정 파일('{filepath}')을 찾을 수 없습니다. 오타 교정 없이 진행합니다.")
    return correction_dict

def post_process_text(ocr_result, wrapper, correction_dict):
    paragraphs = []
    current_paragraph = ""
    for item in ocr_result:
        line = item[1].strip()
        current_paragraph += line + " "
        if line.endswith(('.', '?', '!', ':')):
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    corrected_paragraphs = [p.replace(err, corr) for p in paragraphs for err, corr in correction_dict.items()]
    wrapper.width = main_config.get('text_wrap_width', 70)
    wrapped_paragraphs = [wrapper.fill(p) for p in corrected_paragraphs]
    return "\n\n".join(wrapped_paragraphs)

def ocr_worker(job_q, result_q, reader, wrapper, correction_dict):
    while True:
        try:
            frame = job_q.get(timeout=1)
            if frame is None: break
            try:
                result = reader.readtext(frame)
                if not result:
                    result_q.put("[No text found]")
                    continue
                formatted_text = post_process_text(result, wrapper, correction_dict)
                result_q.put(formatted_text)
            except Exception as e:
                print(f"EasyOCR 작업 중 오류 발생: {e}")
                result_q.put(f"[OCR Error: {e}]")
            finally:
                job_q.task_done()
        except queue.Empty:
            if not is_running:
                break

class SettingsDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("설정")
        self.config = config.copy()
        self.layout = QFormLayout(self)
        self.widgets = {}
        descriptions = {
            'ocr_languages': "OCR 언어 (쉼표로 구분, 예: ko,en) *재시작 필요",
            'gpu_enabled': "GPU 가속 사용 여부 *재시작 필요",
            'motion_threshold': "이 값보다 큰 움직임(Difference)이 감지되면 'Flipping' 상태로 전환",
            'stabilization_delay_seconds': "움직임이 멈춘 후, 캡처를 실행하기까지 대기하는 시간 (초)",
            'stability_threshold_frames': "움직임이 멈춘 상태가 몇 프레임 동안 유지되어야 안정된 것으로 판단할지 결정",
            'user_cooldown_seconds': "캡처 성공 후, 다음 움직임을 감지하기까지의 최소 대기 시간 (초)",
            'text_wrap_width': "결과 텍스트의 한 줄 최대 글자 수"
        }
        for key, value in self.config.items():
            frame = QWidget()
            row_layout = QVBoxLayout(frame)
            row_layout.setContentsMargins(0,0,0,5)
            if isinstance(value, bool):
                widget = QCheckBox(key)
                widget.setChecked(value)
            else:
                widget = QLineEdit()
                widget.setPlaceholderText(key)
                if isinstance(value, list):
                    widget.setText(",".join(value))
                else:
                    widget.setText(str(value))
            desc_label = QPushButton(descriptions.get(key, ""), enabled=False)
            desc_label.setStyleSheet("Text-align:left; border:0; color:grey;")
            row_layout.addWidget(widget)
            row_layout.addWidget(desc_label)
            self.widgets[key] = widget
            self.layout.addRow(frame)
        self.buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttonBox.accepted.connect(self.on_save)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

    def on_save(self):
        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox):
                self.config[key] = widget.isChecked()
            elif key == 'ocr_languages':
                self.config[key] = [lang.strip() for lang in widget.text().split(',')]
            else:
                try: self.config[key] = int(widget.text())
                except ValueError: self.config[key] = float(widget.text())
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=4)
        QMessageBox.information(self, "저장 완료", "설정이 저장되었습니다. 일부 설정은 프로그램을 재시작해야 적용됩니다.")
        super().accept()

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("OCR 결과")
        self.setGeometry(100, 100, 700, 500)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        self.text_area.setFontPointSize(14)
        layout.addWidget(self.text_area)
        button_layout = QHBoxLayout()
        self.copy_button = QPushButton("클립보드로 복사")
        self.clear_button = QPushButton("내용 지우기")
        self.save_as_button = QPushButton("파일로 저장")
        self.settings_button = QPushButton("설정")
        button_layout.addWidget(self.copy_button); button_layout.addWidget(self.clear_button); button_layout.addWidget(self.save_as_button);
        button_layout.addStretch(); button_layout.addWidget(self.settings_button)
        layout.addLayout(button_layout)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.clear_button.clicked.connect(self.clear_text)
        self.save_as_button.clicked.connect(self.save_as)
        self.settings_button.clicked.connect(self.open_settings)

    def add_text(self, text):
        self.text_area.append(text)
        self.text_area.verticalScrollBar().setValue(self.text_area.verticalScrollBar().maximum())

    def copy_to_clipboard(self):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.text_area.toPlainText()); print("UI 창의 내용이 클립보드에 복사되었습니다.")

    def clear_text(self):
        self.text_area.clear()

    def save_as(self):
        text_to_save = self.text_area.toPlainText()
        if not text_to_save: QMessageBox.warning(self, "경고", "저장할 내용이 없습니다."); return
        path, _ = QFileDialog.getSaveFileName(self, "파일로 저장", "output.txt", "Text Files (*.txt)")
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f: f.write(text_to_save)
                QMessageBox.information(self, "성공", f"파일이 성공적으로 저장되었습니다: {path}")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"파일 저장 중 오류가 발생했습니다: {e}")

    def open_settings(self):
        global main_config
        dialog = SettingsDialog(main_config, self)
        if dialog.exec():
            main_config = dialog.config
            print("설정 변경이 감지되었습니다. 적용을 위해 프로그램을 재시작하세요.")

    def closeEvent(self, event):
        global is_running
        is_running = False
        event.accept()

def main():
    global is_running, main_config
    mode = ""
    while mode not in ['1', '2']:
        mode = input("어떤 모드로 실행하시겠습니까? (1: 실시간 OCR, 2: 이미지 저장): ")
    mode = 'ocr' if mode == '1' else 'image'
    print(f"{mode.upper()} 모드로 실행합니다.")

    try:
        with open("config.json", "r", encoding="utf-8") as f: main_config = json.load(f)
    except FileNotFoundError:
        main_config = {
            "ocr_languages": ["ko", "en"], "gpu_enabled": True, "motion_threshold": 0.1,
            "stabilization_delay_seconds": 0.5, "stability_threshold_frames": 5,
            "user_cooldown_seconds": 0.4, "text_wrap_width": 70
        }

    cap = find_capture_device()
    if cap is None: sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"현재 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    ret, first_frame = cap.read()
    if not ret: print("오류: 카메라 프레임을 읽을 수 없습니다."); sys.exit(1)
    
    roi_window_name = "ROI 선택 (마우스로 영역 지정 후 Enter, 취소는 c)"
    roi = cv2.selectROI(roi_window_name, first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(roi_window_name)
    roi_x, roi_y, roi_w, roi_h = [int(c) for c in roi]
    if roi_w == 0 or roi_h == 0: print("경고: ROI가 선택되지 않았습니다. 전체 화면으로 진행합니다."); roi_x, roi_y, roi_w, roi_h = 0, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"관심 영역 선택 완료: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")

    app = QApplication(sys.argv)
    main_window = None
    if mode == 'ocr':
        print("EasyOCR 모델을 로드하는 중입니다...")
        reader = easyocr.Reader(main_config['ocr_languages'], gpu=main_config['gpu_enabled'])
        print(f"EasyOCR 모델 로드 완료. [실행 장치: {reader.device}]")
        correction_dict = load_corrections("corrections.txt")
        wrapper = textwrap.TextWrapper(width=main_config['text_wrap_width'], break_long_words=False, replace_whitespace=False)
        job_queue = queue.Queue(); result_queue = queue.Queue()
        ocr_thread = threading.Thread(target=ocr_worker, args=(job_queue, result_queue, reader, wrapper, correction_dict), daemon=True); ocr_thread.start()
        main_window = MainWindow(main_config)
        main_window.show()
    else: # Image mode
        saved_image_paths = []
        if not os.path.exists('captures'): os.makedirs('captures')

    window_title = "OCR Application"; cv2.namedWindow(window_title)

    status, is_flipping, mean_diff, last_capture_time, stabilizing_since = "Ready", False, 0.0, 0, None
    page_counter, previous_frame_gray = 0, None

    STATUS_COLORS = {
        "Ready": (0, 255, 0), "Flipping...": (0, 255, 255), "Stabilizing...": (255, 255, 0),
        "OCR Queued": (255, 0, 0), "Saved!": (255, 0, 255), "Image Saved!": (0, 165, 255)
    }

    try:
        with open("output.txt", "a", encoding="utf-8") as output_file:
            while is_running:
                ret, frame = cap.read()
                if not ret: break

                cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)

                if mode == 'ocr':
                    try:
                        ocr_text = result_queue.get_nowait()
                        page_counter += 1
                        ui_output = f"--- Page {page_counter} ---\n{ocr_text}"
                        main_window.add_text(ui_output)
                        output_file.write(ocr_text + "\n\n"); output_file.flush()
                        print(f"Page {page_counter} processed and saved.")
                        last_capture_time = time.time(); status = "Saved!"
                        result_queue.task_done()
                    except queue.Empty:
                        pass

                roi_frame_for_diff = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                gray = cv2.cvtColor(roi_frame_for_diff, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if previous_frame_gray is not None:
                    diff = cv2.absdiff(previous_frame_gray, gray); mean_diff = np.mean(diff)

                    if mean_diff > main_config['motion_threshold'] and not is_flipping and (time.time() - last_capture_time > main_config['user_cooldown_seconds']):
                        status = "Flipping..."; is_flipping = True; stabilizing_since = None

                    if is_flipping:
                        if mean_diff == 0.0:
                            if stabilizing_since is None: stabilizing_since = time.time(); status = "Stabilizing..."
                        else: stabilizing_since = None; status = "Flipping..."
                        if stabilizing_since is not None and (time.time() - stabilizing_since > main_config['stabilization_delay_seconds']):
                            roi_to_capture = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                            if mode == 'ocr':
                                job_queue.put(roi_to_capture.copy()); status = "OCR Queued"
                            else: # Image mode
                                page_counter += 1
                                filename = f"captures/capture_{page_counter:04d}.png"
                                cv2.imwrite(filename, roi_to_capture)
                                saved_image_paths.append(filename)
                                print(f"Image {filename} saved.")
                                status = "Image Saved!"
                                last_capture_time = time.time()
                            is_flipping = False; stabilizing_since = None

                    current_status_key = status.split('!')[0] + '!' if '!' in status else status
                    if not is_flipping and current_status_key not in ["Saved!", "OCR Queued", "Image Saved!"]: status = "Ready"
                    elif status in ["Saved!", "Image Saved!"] and (time.time() - last_capture_time > main_config['user_cooldown_seconds']):
                        status = "Ready"

                cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, STATUS_COLORS.get(status, (0,0,255)), 2, cv2.LINE_AA)
                cv2.imshow(window_title, frame)
                previous_frame_gray = gray.copy()

                if mode == 'ocr': app.processEvents()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): is_running = False

    finally:
        print("프로그램 종료 중...")
        if mode == 'ocr':
            job_queue.put(None)
            ocr_thread.join(timeout=5)
        elif mode == 'image' and saved_image_paths:
            print(f"{len(saved_image_paths)}개의 이미지를 PDF로 변환합니다...")
            try:
                pil_images = [Image.open(p).convert('RGB') for p in saved_image_paths]
                if pil_images:
                    pil_images[0].save("output.pdf", save_all=True, append_images=pil_images[1:])
                    print("output.pdf 파일이 성공적으로 생성되었습니다.")
            except Exception as e:
                print(f"PDF 생성 중 오류 발생: {e}")

        cap.release(); cv2.destroyAllWindows()
        if mode == 'ocr': app.quit()
        print("모든 리소스를 해제하고 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()