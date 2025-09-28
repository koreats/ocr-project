import os
import re
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QLineEdit, QPushButton, QMessageBox
from PyQt6.QtCore import Qt

# Backend functions from Phase 1
def get_projects(output_dir):
    """Scans the output directory and returns a list of project names."""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            return [] # Return empty if we can't even create it
    try:
        return sorted([d.name for d in os.scandir(output_dir) if d.is_dir()])
    except OSError:
        return []

def get_project_sessions(project_path):
    """Scans a project directory and returns a sorted list of session names."""
    if not os.path.exists(project_path):
        return []
    try:
        sessions = [d.name for d in os.scandir(project_path) if d.is_dir()]
        sessions.sort(reverse=True)
        return sessions
    except OSError:
        return []

def load_session_data(session_path):
    """Loads the data (OCR text, image files) from a session folder."""
    data = {'ocr_text': '', 'image_files': []}
    if not os.path.exists(session_path):
        return data

    ocr_file = os.path.join(session_path, 'ocr_results.txt')
    if os.path.exists(ocr_file):
        try:
            with open(ocr_file, 'r', encoding='utf-8') as f:
                data['ocr_text'] = f.read()
        except (IOError, OSError):
            pass

    captures_dir = os.path.join(session_path, 'captures')
    if os.path.exists(captures_dir):
        try:
            images = [os.path.join(captures_dir, f) for f in os.listdir(captures_dir) if f.endswith('.png')]
            images.sort()
            data['image_files'] = images
        except OSError:
            pass
            
    return data

class ProjectManagerDialog(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.output_dir = self.config.get('output_directory', 'output')
        self.project_path = None

        self.setWindowTitle("프로젝트 관리자")
        self.setMinimumWidth(400)

        # --- Layouts ---
        self.main_layout = QVBoxLayout(self)
        self.list_layout = QVBoxLayout()
        self.new_proj_layout = QHBoxLayout()
        self.actions_layout = QHBoxLayout()

        # --- Widgets ---
        self.list_label = QLabel("기존 프로젝트 선택:")
        self.project_list_widget = QListWidget()
        self.new_proj_label = QLabel("새 프로젝트 이름:")
        self.new_proj_input = QLineEdit()
        self.create_button = QPushButton("새로 만들기")
        self.load_button = QPushButton("선택 프로젝트로 시작")
        self.quick_capture_button = QPushButton("빠른 캡처")

        # --- Setup UI ---
        self.list_layout.addWidget(self.list_label)
        self.list_layout.addWidget(self.project_list_widget)

        self.new_proj_layout.addWidget(self.new_proj_label)
        self.new_proj_layout.addWidget(self.new_proj_input)
        self.new_proj_layout.addWidget(self.create_button)

        self.actions_layout.addStretch()
        self.actions_layout.addWidget(self.quick_capture_button)
        self.actions_layout.addWidget(self.load_button)
        self.actions_layout.addStretch()

        self.main_layout.addLayout(self.list_layout)
        self.main_layout.addSpacing(15)
        self.main_layout.addLayout(self.new_proj_layout)
        self.main_layout.addSpacing(15)
        self.main_layout.addLayout(self.actions_layout)

        # --- Connections ---
        self.create_button.clicked.connect(self._create_project)
        self.load_button.clicked.connect(self._load_project)
        self.quick_capture_button.clicked.connect(self._quick_capture)
        self.project_list_widget.itemDoubleClicked.connect(self._load_project)
        self.new_proj_input.returnPressed.connect(self._create_project)

        # --- Initial Population ---
        self._populate_projects()

    def _populate_projects(self):
        self.project_list_widget.clear()
        projects = get_projects(self.output_dir)
        self.project_list_widget.addItems(projects)
        if self.project_list_widget.count() > 0:
            self.project_list_widget.setCurrentRow(0)

    def _validate_project_name(self, name):
        if not name or not name.strip():
            QMessageBox.warning(self, "입력 오류", "프로젝트 이름은 공백일 수 없습니다.")
            return False
        # Basic validation for invalid filesystem characters
        if not re.match(r'^[a-zA-Z0-9_\-가-힣 ]+', name):
            QMessageBox.warning(self, "입력 오류", "프로젝트 이름에 특수문자 (\ / : * ? \" < > |)는 사용할 수 없습니다.")
            return False
        return True

    def _create_project(self):
        project_name = self.new_proj_input.text().strip()
        if not self._validate_project_name(project_name):
            return

        project_path = os.path.join(self.output_dir, project_name)
        if os.path.exists(project_path):
            QMessageBox.warning(self, "생성 오류", f"'{project_name}' 프로젝트가 이미 존재합니다.")
            return
        
        try:
            os.makedirs(project_path)
            self.project_path = project_path
            self.accept()
        except (OSError, PermissionError) as e:
            QMessageBox.critical(self, "폴더 생성 오류", f"프로젝트 폴더를 생성할 수 없습니다: {e}")

    def _load_project(self):
        selected_item = self.project_list_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "선택 오류", "먼저 프로젝트를 선택하세요.")
            return
        
        project_name = selected_item.text()
        self.project_path = os.path.join(self.output_dir, project_name)
        self.accept()

    def _quick_capture(self):
        project_name = "_QuickCapture"
        self.project_path = os.path.join(self.output_dir, project_name)
        # Ensure the directory exists
        if not os.path.exists(self.project_path):
            try:
                os.makedirs(self.project_path)
            except (OSError, PermissionError) as e:
                QMessageBox.critical(self, "폴더 생성 오류", f"빠른 캡처 폴더를 생성할 수 없습니다: {e}")
                return
        self.accept()
