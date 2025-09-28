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
from datetime import datetime

from PyQt6.QtWidgets import QApplication, QMessageBox

# Local imports
from main_window import MainWindow
from utils import load_corrections, post_process_text

is_running = True
main_config = {}

def find_capture_device():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened(): return cap
    return None



def ocr_worker(job_q, result_q, reader, wrapper, correction_dict, config):
    """Worker thread for performing OCR."""
    while True:
        try:
            frame = job_q.get(timeout=1)
        except queue.Empty:
            if not is_running:
                break
            continue
        
        try:
            if frame is None:
                break
            result = reader.readtext(frame)
            formatted_text = post_process_text(result, wrapper, correction_dict, config) if result else "[No text found]"
            result_q.put(formatted_text)
        except Exception as e:
            result_q.put(f"[OCR Error: {e}]")
        finally:
            job_q.task_done()







def live_capture_mode(config, app):
    global is_running, main_config
    is_running = True
    main_config = config
    main_window = MainWindow(config)
    main_window.show()

    # --- Session Setup ---
    try:
        output_dir = config.get('output_directory', 'output')
        session_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_path = os.path.join(output_dir, session_name)
        captures_path = os.path.join(session_path, 'captures')
        os.makedirs(captures_path, exist_ok=True)
        main_window.session_path = session_path
        main_window.add_log(f"세션을 시작합니다. 결과는 '{session_path}' 폴더에 저장됩니다.")
    except (IOError, PermissionError) as e:
        QMessageBox.critical(main_window, "폴더 생성 오류", f"세션 폴더를 생성할 수 없습니다: {e}\n프로그램을 종료합니다.")
        return

    # --- OCR Reader Pre-loading ---
    main_window.video_label.setText("EasyOCR 모델을 로드하는 중입니다...")
    app.processEvents()
    try:
        reader = easyocr.Reader(config.get('ocr_languages', ['ko', 'en']), gpu=config.get('gpu_enabled', True))
        main_window.add_log(f"EasyOCR 모델 로드 완료. [실행 장치: {reader.device}]")
    except Exception as e:
        main_window.add_log(f"EasyOCR 모델 로드 실패: {e}")
        QMessageBox.critical(main_window, "오류", f"EasyOCR 모델 로드에 실패했습니다. 프로그램을 종료합니다.\n{e}")
        return
    main_window.video_label.setText("카메라 로딩 중...")
    app.processEvents()
    # ---

    main_window.add_log("실시간 캡처 모드로 시작합니다.")
    cap = find_capture_device()
    if cap is None: main_window.add_log("오류: 캡처 장치를 찾을 수 없습니다."); return
    main_window.add_log(f"캡처 장치 로드 완료.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    correction_dict = load_corrections("corrections.txt"); main_window.add_log(f"{len(correction_dict)}개의 교정 규칙을 로드했습니다.")
    wrapper = textwrap.TextWrapper(width=config.get('text_wrap_width', 70), break_long_words=False, replace_whitespace=False)
    job_queue = queue.Queue(); result_queue = queue.Queue()
    ocr_thread = threading.Thread(target=ocr_worker, args=(job_queue, result_queue, reader, wrapper, correction_dict, config), daemon=True); ocr_thread.start()
    
    status, is_flipping, mean_diff, last_capture_time = "Ready", False, 0.0, 0
    page_counter, previous_frame_gray, stability_counter = 0, None, 0
    STATUS_COLORS = { "Ready": (0, 255, 0), "Flipping...": (0, 255, 255), "Stabilizing...": (255, 255, 0), "OCR Queued": (255, 0, 0), "Saved!": (255, 0, 255), "Image Saved!": (0, 165, 255) }
    
    roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
    is_roi_confirmed = False

    try:
        while is_running:
            if not main_window.isVisible(): is_running = False; continue
            ret, frame = cap.read();
            if not ret: break
            display_frame = frame.copy()

            if main_window.current_state == 'ROI_SELECTION':
                status = "ROI Selection"
            
            elif main_window.current_state == 'CAPTURE_MODE' and not is_roi_confirmed:
                selection_rect = main_window.video_label.selection_rect
                if selection_rect.width() > 0 and selection_rect.height() > 0:
                    label_size = main_window.video_label.size()
                    frame_h, frame_w = frame.shape[:2]
                    label_ratio = label_size.width() / label_size.height()
                    frame_ratio = frame_w / frame_h
                    if label_ratio > frame_ratio:
                        scaled_h = label_size.height()
                        scaled_w = int(scaled_h * frame_ratio)
                        x_offset = (label_size.width() - scaled_w) / 2; y_offset = 0
                    else:
                        scaled_w = label_size.width()
                        scaled_h = int(scaled_w / frame_ratio)
                        x_offset = 0; y_offset = (label_size.height() - scaled_h) / 2
                    scale_w_ratio = frame_w / scaled_w; scale_h_ratio = frame_h / scaled_h
                    roi_x = int((selection_rect.x() - x_offset) * scale_w_ratio)
                    roi_y = int((selection_rect.y() - y_offset) * scale_h_ratio)
                    roi_w = int(selection_rect.width() * scale_w_ratio)
                    roi_h = int(selection_rect.height() * scale_h_ratio)
                    roi_x = max(0, roi_x); roi_y = max(0, roi_y)
                    if roi_x + roi_w > frame_w: roi_w = frame_w - roi_x
                    if roi_y + roi_h > frame_h: roi_h = frame_h - roi_y
                    main_window.add_log(f"최종 ROI 확정: x={roi_x}, y={roi_y}, w={roi_w}, h={roi_h}")
                else:
                    roi_x, roi_y, roi_w, roi_h = 0, 0, frame.shape[1], frame.shape[0]
                    main_window.add_log("경고: ROI가 선택되지 않아 전체 화면으로 진행합니다.")
                is_roi_confirmed = True

            if is_roi_confirmed:
                cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
                clean_roi_frame = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()

                if not result_queue.empty():
                    try:
                        ocr_text = result_queue.get_nowait()
                        if main_window.ocr_radio.isChecked():
                            page_counter += 1
                            ui_output = f"--- Page {page_counter} ---\n{ocr_text}"
                            main_window.add_ocr_text(ui_output)
                            try:
                                ocr_file_path = os.path.join(session_path, "ocr_results.txt")
                                with open(ocr_file_path, "a", encoding="utf-8") as f:
                                    f.write(ocr_text + "\n\n")
                                main_window.add_log(f"Page {page_counter} 처리 및 파일에 저장 완료.")
                                status = "Saved!"
                            except (IOError, PermissionError) as e:
                                main_window.add_log(f"OCR 파일 저장 오류: {e}")
                        last_capture_time = time.time()
                        result_queue.task_done()
                    except queue.Empty: pass

                gray = cv2.cvtColor(clean_roi_frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
                if previous_frame_gray is not None:
                    diff = cv2.absdiff(previous_frame_gray, gray); mean_diff = np.mean(diff)
                    if mean_diff > config.get('motion_threshold', 1.5) and not is_flipping and (time.time() - last_capture_time > config.get('user_cooldown_seconds', 0.4)):
                        status = "Flipping..."; is_flipping = True
                    if is_flipping:
                        if mean_diff <= config.get('motion_threshold', 1.5):
                            stability_counter += 1; status = "Stabilizing..."
                        else:
                            stability_counter = 0; status = "Flipping..."
                        if stability_counter >= config.get('stability_threshold_frames', 5):
                            if main_window.ocr_radio.isChecked():
                                job_queue.put(clean_roi_frame.copy()); status = "OCR Queued"
                            else: # Scan mode
                                try:
                                    img_count = len(main_window.saved_image_paths) + 1
                                    filename = os.path.join(captures_path, f"{img_count:04d}.png")
                                    cv2.imwrite(filename, clean_roi_frame)
                                    main_window.saved_image_paths.append(filename)
                                    main_window.ocr_results_area.setText("\n".join(main_window.saved_image_paths))
                                    main_window.add_log(f"이미지 저장됨: {filename}")
                                    scan_radio_text = f"문서 스캔 (PDF 생성) ({img_count}장 캡처됨)"
                                    main_window.scan_radio.setText(scan_radio_text)
                                    status = "Image Saved!"; last_capture_time = time.time()
                                except (IOError, PermissionError) as e:
                                    main_window.add_log(f"이미지 저장 오류: {e}")
                            is_flipping = False; stability_counter = 0
                previous_frame_gray = gray.copy()

            current_status_key = status.split('!')[0] + '!' if '!' in status else status
            if not is_flipping and current_status_key not in ["Saved!", "OCR Queued", "Image Saved!", "ROI Selection"]: status = "Ready"
            elif status in ["Saved!", "Image Saved!"] and (time.time() - last_capture_time > config.get('user_cooldown_seconds', 0.4)): status = "Ready"
            
            status_color = STATUS_COLORS.get(status, (0,0,255))
            cv2.putText(display_frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
            if is_roi_confirmed:
                cv2.putText(display_frame, f"Difference: {mean_diff:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, f"Stability: {stability_counter}/{config.get('stability_threshold_frames', 5)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            main_window.update_video_frame(display_frame)
            app.processEvents()
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]: is_running = False
    finally:
        main_window.add_log("프로그램 종료 중...")
        if job_queue is not None:
            job_queue.put(None)
            ocr_thread.join(timeout=5)
        cap.release()
        if 'app' in locals(): app.quit()

def main():
    global is_running, main_config
    app = QApplication(sys.argv)
    try:
        with open("config.json", "r", encoding="utf-8") as f: main_config = json.load(f)
    except FileNotFoundError:
        main_config = { 
            "ocr_languages": ["ko", "en"], "gpu_enabled": True, "motion_threshold": 1.5, 
            "stabilization_delay_seconds": 0.5, "stability_threshold_frames": 5, 
            "user_cooldown_seconds": 0.4, "text_wrap_width": 70, "output_directory": "output"
        }

    live_capture_mode(main_config, app)

if __name__ == "__main__":
    main()
