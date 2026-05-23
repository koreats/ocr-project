# OCR Project

Python 기반의 **광학 문자 인식(OCR) 애플리케이션**입니다. 실시간 비디오 캡처, 화면 안정성 감지, OCR 처리 및 PDF 생성 기능을 제공합니다.

## 🎯 프로젝트 개요

이 프로젝트는 다음과 같은 기능을 제공합니다:

- **실시간 비디오 캡처**: 연결된 카메라로부터 영상 입력
- **동작 감지**: 화면이 안정적일 때만 OCR 실행
- **멀티스레딩**: 별도 스레드에서 OCR 처리하여 부드러운 UI 제공
- **이미지 전처리**: OCR 정확도 향상
- **EasyOCR 통합**: 한글/영문 고품질 텍스트 인식
- **GUI 인터페이스**: PyQt6 기반의 사용자 친화적 UI
- **PDF 처리**: PDF 파일의 문서 스캔 및 생성
- **프로젝트 관리**: 세션 기반의 작업 관리

## 🛠️ 주요 기술

| 기술 | 용도 |
|------|------|
| **Python** | 핵심 프로그래밍 언어 |
| **OpenCV** | 비디오 캡처, 이미지 처리, 동작 감지 |
| **EasyOCR** | OCR 엔진 (한글/영문 지원) |
| **PyQt6** | GUI 애플리케이션 |
| **PyMuPDF** | PDF 처리 |

## 📦 설치 및 실행

### 필수 라이브러리 설치

```bash
pip install opencv-python pyqt6 easyocr numpy Pillow PyMuPDF
```

### 애플리케이션 실행

```bash
python app.py
```

메인 애플리케이션 윈도우가 열리며, 원하는 모드를 선택하여 OCR 프로세스를 시작할 수 있습니다.

## 🏗️ 프로젝트 구조

```
ocr-project/
├── app.py                    # 메인 실행 파일
├── main_window.py            # 메인 GUI 윈도우
├── project_manager.py        # 프로젝트 관리 기능
├── settings_dialog.py        # 설정 다이얼로그
├── utils.py                  # 유틸리티 함수
├── config.json              # 설정 파일
├── corrections.txt          # OCR 오류 수정 규칙
├── GEMINI.md               # 상세 프로젝트 문서
└── temp/                    # 임시 파일 디렉토리
```

## ⚙️ 설정 (config.json)

```json
{
    "ocr_languages": ["ko", "en"],           // OCR 언어 설정
    "gpu_enabled": true,                     // GPU 가속 여부
    "motion_threshold": 0.1,                 // 동작 감지 임계값
    "stabilization_delay_seconds": 0.5,      // 안정화 지연 시간
    "stability_threshold_frames": 5,         // 안정성 판단 프레임 수
    "user_cooldown_seconds": 0.4,            // 사용자 쿨다운 시간
    "text_wrap_width": 70,                   // 텍스트 래핑 너비
    "output_directory": "output"             // 출력 디렉토리
}
```

## 🚀 사용 모드

### 1. **실시간 OCR 모드**
- 비디오 피드에서 안정적인 화면을 자동 감지
- 감지된 영상에 대해 실시간 OCR 수행
- 결과를 `ocr_results.txt`에 자동 저장

### 2. **문서 스캔 모드 (PDF 생성)**
- 여러 페이지를 이미지로 캡처
- 캡처된 이미지들로부터 PDF 문서 생성
- 스캔 품질 최적화

## 💡 개발 규칙

- 코드는 명확한 주석과 함께 작성됨
- `config.json`을 통한 설정 관리로 코드 수정 없이 동작 변경 가능
- `corrections.txt`를 이용한 OCR 오류 자동 수정
- 각 모듈은 독립적으로 테스트 가능하도록 설계

## 📝 라이선스

이 프로젝트는 자유롭게 사용, 수정, 배포할 수 있습니다.

## 🤝 기여

버그 리포트 및 기능 제안은 Issues를 통해 제출해주세요.
