import cv2
import sys
import numpy as np
import threading
import queue
import easyocr
import time
import textwrap
import tkinter as tk
from tkinter import scrolledtext, messagebox
import json

is_running = True

def find_capture_device():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
    return None

def post_process_text(ocr_result, wrapper):
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
    wrapped_paragraphs = [wrapper.fill(p) for p in paragraphs]
    return "\n\n".join(wrapped_paragraphs)

def ocr_worker(job_q, result_q, reader, wrapper):
    while True:
        try:
            frame = job_q.get(timeout=1)
            if frame is None: break
            try:
                result = reader.readtext(frame)
                if not result:
                    result_q.put("[No text found]")
                    continue
                formatted_text = post_process_text(result, wrapper)
                result_q.put(formatted_text)
            except Exception as e:
                print(f"OCR 작업 중 오류 발생: {e}")
                result_q.put(f"[OCR Error: {e}]")
            finally:
                job_q.task_done()
        except queue.Empty:
            if not is_running:
                break

def on_key_press(event):
    global is_running
    if event.keysym == 'Escape' or event.keysym == 'q':
        is_running = False

def open_settings_window(config, root):
    settings_window = tk.Toplevel(root)
    settings_window.title("설정")
    settings_window.geometry("550x350")

    vars = {
        'ocr_languages': tk.StringVar(value=",".join(config.get('ocr_languages', ['ko', 'en']))),
        'gpu_enabled': tk.BooleanVar(value=config.get('gpu_enabled', True)),
        'motion_threshold': tk.DoubleVar(value=config.get('motion_threshold', 0.1)),
        'stabilization_delay_seconds': tk.DoubleVar(value=config.get('stabilization_delay_seconds', 0.5)),
        'stability_threshold_frames': tk.IntVar(value=config.get('stability_threshold_frames', 5)),
        'user_cooldown_seconds': tk.DoubleVar(value=config.get('user_cooldown_seconds', 0.4)),
        'text_wrap_width': tk.IntVar(value=config.get('text_wrap_width', 70))
    }

    descriptions = {
        'ocr_languages': "OCR 언어 (쉼표로 구분, 예: ko,en) *재시작 필요",
        'gpu_enabled': "GPU 가속 사용 여부 *재시작 필요",
        'motion_threshold': "이 값보다 큰 움직임(Difference)이 감지되면 'Flipping' 상태로 전환",
        'stabilization_delay_seconds': "움직임이 멈춘 후, 캡처를 실행하기까지 대기하는 시간 (초)",
        'stability_threshold_frames': "움직임이 멈춘 상태가 몇 프레임 동안 유지되어야 안정된 것으로 판단할지 결정",
        'user_cooldown_seconds': "캡처 성공 후, 다음 움직임을 감지하기까지의 최소 대기 시간 (초)",
        'text_wrap_width': "결과 텍스트의 한 줄 최대 글자 수"
    }

    # Create UI elements
    for i, (key, var) in enumerate(vars.items()):
        frame = tk.Frame(settings_window)
        frame.pack(fill='x', padx=10, pady=5)
        label = tk.Label(frame, text=key, width=25, anchor='w')
        label.pack(side='left')
        if isinstance(var, tk.BooleanVar):
            widget = tk.Checkbutton(frame, variable=var)
        else:
            widget = tk.Entry(frame, textvariable=var, width=10)
        widget.pack(side='left', padx=5)
        desc_label = tk.Label(frame, text=descriptions[key], fg="grey", anchor='w')
        desc_label.pack(side='left')

    def save_settings():
        new_config = {}
        for key, var in vars.items():
            if key == 'ocr_languages':
                new_config[key] = [lang.strip() for lang in var.get().split(',')]
            else:
                new_config[key] = var.get()
        
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(new_config, f, indent=4)
        
        messagebox.showinfo("저장 완료", "설정이 저장되었습니다. 일부 설정은 프로그램을 재시작해야 적용됩니다.")
        settings_window.destroy()

    save_button = tk.Button(settings_window, text="저장 및 닫기", command=save_settings)
    save_button.pack(pady=10)

def main():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {
            "ocr_languages": ["ko", "en"], "gpu_enabled": True, "motion_threshold": 0.1,
            "stabilization_delay_seconds": 0.5, "stability_threshold_frames": 5,
            "user_cooldown_seconds": 0.4, "text_wrap_width": 70
        }

    cap = find_capture_device()
    if cap is None: sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"현재 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    print("EasyOCR 모델을 로드하는 중입니다...")
    reader = easyocr.Reader(config['ocr_languages'], gpu=config['gpu_enabled'])
    print(f"EasyOCR 모델 로드 완료. [실행 장치: {reader.device}]")

    wrapper = textwrap.TextWrapper(width=config['text_wrap_width'], break_long_words=False, replace_whitespace=False)
    job_queue = queue.Queue(); result_queue = queue.Queue()
    ocr_thread = threading.Thread(target=ocr_worker, args=(job_queue, result_queue, reader, wrapper), daemon=True); ocr_thread.start()

    # --- Tkinter UI Setup ---
    root = tk.Tk(); root.title("OCR 결과")
    root.geometry("600x400")

    # Create a bottom frame for the button
    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    # Create a top frame for the text area
    top_frame = tk.Frame(root)
    top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Add the text area to the top frame
    text_area = scrolledtext.ScrolledText(top_frame, wrap=tk.WORD, font=("Helvetica", 14))
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Add the button to the bottom frame
    settings_button = tk.Button(bottom_frame, text="설정", command=lambda: open_settings_window(config, root))
    settings_button.pack()

    root.bind("<KeyPress>", on_key_press)
    # ------------------------

    window_title = "OCR Application"; cv2.namedWindow(window_title)

    status, is_flipping, mean_diff, last_capture_time, stabilizing_since = "Ready", False, 0.0, 0, None
    page_counter, previous_frame_gray = 0, None

    STATUS_COLORS = {
        "Ready": (0, 255, 0), "Flipping...": (0, 255, 255), "Stabilizing...": (255, 255, 0),
        "OCR Queued": (255, 0, 0), "Saved!": (255, 0, 255)
    }

    try:
        with open("output.txt", "a", encoding="utf-8") as output_file:
            while is_running:
                ret, frame = cap.read()
                if not ret: break

                try:
                    ocr_text = result_queue.get_nowait()
                    page_counter += 1
                    ui_output = f"--- Page {page_counter} ---\n{ocr_text}\n\n"
                    text_area.insert(tk.END, ui_output); text_area.see(tk.END)
                    output_file.write(ocr_text + "\n\n"); output_file.flush()
                    print(f"Page {page_counter} processed and saved.")
                    last_capture_time = time.time(); status = "Saved!"
                    result_queue.task_done()
                except queue.Empty:
                    pass

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if previous_frame_gray is not None:
                    diff = cv2.absdiff(previous_frame_gray, gray); mean_diff = np.mean(diff)

                    if mean_diff > config['motion_threshold'] and not is_flipping and (time.time() - last_capture_time > config['user_cooldown_seconds']):
                        status = "Flipping..."; is_flipping = True; stabilizing_since = None

                    if is_flipping:
                        if mean_diff == 0.0:
                            if stabilizing_since is None: stabilizing_since = time.time(); status = "Stabilizing..."
                        else: stabilizing_since = None; status = "Flipping..."
                        if stabilizing_since is not None and (time.time() - stabilizing_since > config['stabilization_delay_seconds']):
                            job_queue.put(frame.copy()); status = "OCR Queued"; is_flipping = False; stabilizing_since = None

                    if not is_flipping and status not in ["Saved!", "OCR Queued"]: status = "Ready"
                    elif status == "Saved!" and (time.time() - last_capture_time > config['user_cooldown_seconds']):
                        status = "Ready"

                status_color = STATUS_COLORS.get(status, (0, 0, 255))
                cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
                cv2.imshow(window_title, frame)
                previous_frame_gray = gray.copy()

                root.update(); root.update_idletasks()
                cv2.waitKey(1)
    finally:
        print("프로그램 종료 중...")
        job_queue.put(None)
        ocr_thread.join(timeout=5)
        cap.release(); cv2.destroyAllWindows()
        try: root.destroy()
        except tk.TclError: pass
        print("모든 리소스를 해제하고 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()