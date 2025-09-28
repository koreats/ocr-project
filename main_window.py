import os
import cv2
import numpy as np
import easyocr
import textwrap
import fitz  # PyMuPDF
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QHBoxLayout, QWidget,
    QDialog, QFormLayout, QLineEdit, QCheckBox, QDialogButtonBox, QMessageBox,
    QFileDialog, QLabel, QRadioButton, QGridLayout
)
from PyQt6.QtGui import QGuiApplication, QImage, QPixmap, QPainter, QPen
from PyQt6.QtCore import Qt, QRect

from settings_dialog import SettingsDialog
from utils import load_corrections, post_process_text

class AspectRatioLabel(QLabel):
    def __init__(self, main_window_ref, parent=None):
        super().__init__(parent)
        self.main_window = main_window_ref
        self.setMinimumSize(1, 1)
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.selection_rect = QRect()

    def mousePressEvent(self, event):
        if self.main_window.current_state != 'ROI_SELECTION': return
        self.is_drawing = True
        self.start_point = event.pos()
        self.selection_rect = QRect(self.start_point, event.pos())
        self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.selection_rect = QRect(self.start_point, event.pos()).normalized()
            self.update() # Trigger a repaint

    def mouseReleaseEvent(self, event):
        if self.is_drawing:
            self.is_drawing = False
            self.selection_rect = QRect(self.start_point, event.pos()).normalized()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.selection_rect.isNull():
            painter = QPainter(self)
            pen = QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)

    def resizeEvent(self, event):
        super().resizeEvent(event)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return int(width * 9 / 16)

class ModeSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("모드 선택")
        self.layout = QVBoxLayout(self)
        self.selected_mode = None
        self.btn_ocr = QPushButton("1. 실시간 텍스트 추출 (카메라 → TXT)")
        self.btn_scan = QPushButton("2. 카메라 스캔 (→ PDF + TXT)")
        self.btn_ocr.clicked.connect(lambda: self.set_mode('ocr'))
        self.btn_scan.clicked.connect(lambda: self.set_mode('scan'))
        self.layout.addWidget(QLabel("실행할 작업 모드를 선택하세요:"))
        self.layout.addWidget(self.btn_ocr)
        self.layout.addWidget(self.btn_scan)

    def set_mode(self, mode):
        self.selected_mode = mode
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.session_path = None # To be set later
        self.pdf_saved_this_session = False
        self.saved_image_paths = []
        self.current_state = 'ROI_SELECTION' # Initial state
        self.setWindowTitle("OCR Application")
        self.setGeometry(100, 100, 1280, 720)

        central_widget = QWidget(); self.setCentralWidget(central_widget)
        grid_layout = QGridLayout(central_widget)

        # --- Left Column Widgets ---
        self.video_label = AspectRatioLabel(self)
        
        # Controls Widget (Mode + Buttons)
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        self.ocr_radio = QRadioButton("실시간 텍스트 추출")
        self.scan_radio = QRadioButton("문서 스캔 (PDF 생성)")
        self.ocr_radio.setChecked(True)
        self.copy_button = QPushButton("클립보드로 복사")
        self.clear_button = QPushButton("내용/목록 지우기")
        self.save_as_button = QPushButton("텍스트 파일로 저장")
        self.finish_scan_button = QPushButton("스캔 완료 및 파일 생성")
        self.settings_button = QPushButton("설정")
        self.confirm_roi_button = QPushButton("영역 확정")

        controls_layout.addWidget(self.ocr_radio)
        controls_layout.addWidget(self.scan_radio)
        controls_layout.addStretch()
        controls_layout.addWidget(self.copy_button)
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.save_as_button)
        controls_layout.addWidget(self.finish_scan_button)
        controls_layout.addWidget(self.settings_button)
        controls_layout.addWidget(self.confirm_roi_button)

        self.log_area = QTextEdit(); self.log_area.setReadOnly(True)

        # --- Right Column Widgets ---
        self.ocr_results_area = QTextEdit(); self.ocr_results_area.setReadOnly(True); self.ocr_results_area.setFontPointSize(14)

        # --- Add Widgets to Grid ---
        grid_layout.addWidget(self.video_label, 0, 0)
        grid_layout.addWidget(controls_widget, 1, 0)
        grid_layout.addWidget(self.log_area, 2, 0)
        grid_layout.addWidget(self.ocr_results_area, 0, 1, 3, 1) # Span 3 rows

        # --- Set Stretches ---
        grid_layout.setColumnStretch(0, 2) # Left column is 2/3 of width
        grid_layout.setColumnStretch(1, 1) # Right column is 1/3 of width
        grid_layout.setRowStretch(0, 0)    # Video row height is driven by aspect ratio
        grid_layout.setRowStretch(1, 0)    # Controls row height is fixed
        grid_layout.setRowStretch(2, 1)    # Log row takes remaining space

        # --- Connections and Initial State ---
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.clear_button.clicked.connect(self.clear_text)
        self.save_as_button.clicked.connect(self.save_as)
        self.finish_scan_button.clicked.connect(self.finish_scan_session)
        self.settings_button.clicked.connect(self.open_settings)
        self.confirm_roi_button.clicked.connect(self._confirm_roi)
        
        self.ocr_radio.toggled.connect(self._on_mode_changed)
        self._update_ui_for_state() # Set initial UI

    def _update_ui_for_state(self):
        if self.current_state == 'ROI_SELECTION':
            self.ocr_radio.setVisible(False)
            self.scan_radio.setVisible(False)
            self.copy_button.setVisible(False)
            self.clear_button.setVisible(False)
            self.save_as_button.setVisible(False)
            self.finish_scan_button.setVisible(False)
            self.settings_button.setVisible(False)
            self.confirm_roi_button.setVisible(True)
        elif self.current_state == 'CAPTURE_MODE':
            self.ocr_radio.setVisible(True)
            self.scan_radio.setVisible(True)
            self.settings_button.setVisible(True)
            self.clear_button.setVisible(True)
            self.confirm_roi_button.setVisible(False)
            self._on_mode_changed() # Update buttons based on radio selection

    def _confirm_roi(self):
        self.current_state = 'CAPTURE_MODE'
        self.video_label.is_drawing = False # Disable further drawing
        if self.video_label.selection_rect.isNull():
            self.add_log("선택된 영역이 없어 전체 화면으로 ROI를 자동 설정합니다.")
        else:
            self.add_log("ROI 확정됨.")
        self._update_ui_for_state()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.current_state == 'ROI_SELECTION' and self.confirm_roi_button.isVisible():
                self._confirm_roi()
                return
        super().keyPressEvent(event)

    def _on_mode_changed(self):
        if self.current_state != 'CAPTURE_MODE':
            return
        self.ocr_results_area.clear()
        if self.ocr_radio.isChecked():
            self.copy_button.setVisible(True)
            self.save_as_button.setVisible(True)
            self.finish_scan_button.setVisible(False)
            self.ocr_results_area.setPlaceholderText("실시간 OCR 결과가 여기에 표시됩니다.")
        else: # Scan mode is checked
            self.copy_button.setVisible(False)
            self.save_as_button.setVisible(False)
            self.finish_scan_button.setVisible(True)
            self.ocr_results_area.setText("\n".join(self.saved_image_paths))
            self.ocr_results_area.setPlaceholderText("캡처된 이미지 목록이 여기에 표시됩니다.")

    def update_video_frame(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def add_ocr_text(self, text):
        self.ocr_results_area.append(text)
        self.ocr_results_area.verticalScrollBar().setValue(self.ocr_results_area.verticalScrollBar().maximum())

    def add_log(self, text):
        self.log_area.append(text)
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    def copy_to_clipboard(self):
        QGuiApplication.clipboard().setText(self.ocr_results_area.toPlainText())
        self.add_log("UI 창의 내용이 클립보드에 복사되었습니다.")

    def clear_text(self):
        self.ocr_results_area.clear()

    def save_as(self):
        text_to_save = self.ocr_results_area.toPlainText()
        if not text_to_save:
            QMessageBox.warning(self, "경고", "저장할 내용이 없습니다.")
            return
        
        default_path = os.path.join(self.session_path, "exported_text.txt") if self.session_path else "exported_text.txt"
        path, _ = QFileDialog.getSaveFileName(self, "텍스트 파일로 저장", default_path, "Text Files (*.txt)")
        
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text_to_save)
                self.add_log(f"파일이 성공적으로 저장되었습니다: {path}")
            except (IOError, PermissionError) as e:
                self.add_log(f"파일 저장 중 오류가 발생했습니다: {e}")
                QMessageBox.critical(self, "파일 저장 오류", f"파일을 저장할 수 없습니다.\n{e}")

    def open_settings(self):
        global main_config
        dialog = SettingsDialog(main_config, self)
        if dialog.exec():
            main_config = dialog.config
            self.add_log("설정 변경이 감지되었습니다. 적용을 위해 프로그램을 재시작하세요.")

    def compile_to_pdf(self):
        self.add_log("PDF 생성을 시작합니다...")
        if not self.saved_image_paths:
            QMessageBox.warning(self, "경고", "PDF로 만들 이미지가 없습니다.")
            return None

        default_path = os.path.join(self.session_path, "scan_session.pdf") if self.session_path else "scan_session.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "PDF로 저장", default_path, "PDF Files (*.pdf)")
        
        if path:
            try:
                pil_images = [Image.open(p).convert('RGB') for p in self.saved_image_paths]
                if pil_images:
                    pil_images[0].save(path, save_all=True, append_images=pil_images[1:])
                    self.add_log(f"PDF 파일이 성공적으로 저장되었습니다: {path}")
                    self.pdf_saved_this_session = True
                    return path
            except (IOError, PermissionError) as e:
                self.add_log(f"PDF 생성 중 오류가 발생했습니다: {e}")
                QMessageBox.critical(self, "PDF 생성 오류", f"PDF 파일을 생성할 수 없습니다.\n{e}")
        return None

    def finish_scan_session(self):
        self.add_log("파일 생성을 시작합니다. 잠시만 기다려주세요...")
        QApplication.processEvents() # Update UI log

        self.finish_scan_button.setEnabled(False)

        try:
            saved_pdf_path = self.compile_to_pdf()
            if saved_pdf_path:
                self.extract_text_from_pdf(saved_pdf_path)
        finally:
            self.add_log("모든 작업 완료. 스캔 상태를 초기화합니다.")
            self.saved_image_paths.clear()
            self.ocr_results_area.clear()
            self.scan_radio.setText("문서 스캔 (PDF 생성)")
            self.finish_scan_button.setEnabled(True)

    def extract_text_from_pdf(self, pdf_path):
        self.add_log(f"'{os.path.basename(pdf_path)}' 파일에서 텍스트 추출을 시작합니다...")
        QApplication.processEvents() # Update UI log
        try:
            reader = easyocr.Reader(self.config.get('ocr_languages', ['ko', 'en']), gpu=self.config.get('gpu_enabled', True))
            self.add_log("EasyOCR 모델 로드 완료 (텍스트 추출용).")
            QApplication.processEvents()

            correction_dict = load_corrections("corrections.txt")
            wrapper = textwrap.TextWrapper(width=self.config.get('text_wrap_width', 70), break_long_words=False, replace_whitespace=False)

            doc = fitz.open(pdf_path)
            full_text = ""
            for i, page in enumerate(doc):
                self.add_log(f"총 {len(doc)} 페이지 중 {i+1} 페이지 처리 중...")
                QApplication.processEvents()
                pix = page.get_pixmap(dpi=300)
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4: img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
                result = reader.readtext(img_data)
                if result: full_text += post_process_text(result, wrapper, correction_dict, self.config) + "\n\n"
            
            output_filename = os.path.splitext(pdf_path)[0] + ".txt"
            with open(output_filename, "w", encoding="utf-8") as f: f.write(full_text)
            
            self.add_log(f"텍스트 추출 완료! 결과가 '{os.path.basename(output_filename)}' 파일에 저장되었습니다.")
            QMessageBox.information(self, "처리 완료", f"PDF에서 텍스트 추출이 완료되었습니다!\n결과가 '{os.path.basename(output_filename)}' 파일에 저장되었습니다.")
        except Exception as e:
            self.add_log(f"PDF 텍스트 추출 중 오류: {e}")
            QMessageBox.critical(self, "오류", f"PDF 텍스트 추출 중 오류가 발생했습니다: {e}")

    def closeEvent(self, event):
        global is_running
        is_running = False
        if self.saved_image_paths and not self.pdf_saved_this_session:
            reply = QMessageBox.question(self, "작업 미완료", f"아직 파일로 생성되지 않은 {len(self.saved_image_paths)}개의 이미지가 있습니다.\n지금 PDF와 텍스트로 변환하시겠습니까?")
            if reply == QMessageBox.StandardButton.Yes:
                self.finish_scan_session()
        event.accept()
