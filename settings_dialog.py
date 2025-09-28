import json
from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QCheckBox, QDialogButtonBox, QMessageBox, QWidget, QVBoxLayout, QPushButton

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
            'text_wrap_width': "결과 텍스트의 한 줄 최대 글자 수",
            'output_directory': "결과물이 저장될 기본 폴더 이름"
        }
        for key, value in self.config.items():
            # Skip keys that are not meant to be user-configurable in this dialog
            if key not in descriptions:
                continue
            frame = QWidget(); row_layout = QVBoxLayout(frame); row_layout.setContentsMargins(0,0,0,5)
            if isinstance(value, bool): 
                widget = QCheckBox(key)
                widget.setChecked(value)
            else:
                widget = QLineEdit()
                widget.setPlaceholderText(key)
                widget.setText(",".join(value) if isinstance(value, list) else str(value))
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
            elif isinstance(self.config[key], (int, float)):
                try:
                    # Try to preserve the original type
                    original_type = type(self.config[key])
                    self.config[key] = original_type(widget.text())
                except (ValueError, TypeError):
                    # Fallback for invalid format
                    self.config[key] = widget.text()
            else:
                 self.config[key] = widget.text()

        try:
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            QMessageBox.information(self, "저장 완료", "설정이 저장되었습니다. 일부 설정은 프로그램을 재시작해야 적용됩니다.")
            super().accept()
        except (IOError, PermissionError) as e:
            QMessageBox.critical(self, "저장 실패", f"설정 파일을 저장하는 데 실패했습니다: {e}")

