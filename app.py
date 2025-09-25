import cv2
import sys
import numpy as np
import threading
import queue
import easyocr
import time
import textwrap
import tkinter as tk
from tkinter import scrolledtext

# --- Global flag for shutdown ---
is_running = True
# --------------------------------

def find_capture_device():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
    return None

def post_process_text(ocr_result):
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
    wrapper = textwrap.TextWrapper(width=70, break_long_words=False, replace_whitespace=False)
    wrapped_paragraphs = [wrapper.fill(p) for p in paragraphs]
    return "\n\n".join(wrapped_paragraphs)

def ocr_worker(job_q, result_q, reader):
    while True:
        try:
            frame = job_q.get(timeout=1) # Use timeout to prevent blocking forever
            if frame is None: break
            try:
                result = reader.readtext(frame)
                if not result:
                    result_q.put("[No text found]")
                    continue
                formatted_text = post_process_text(result)
                result_q.put(formatted_text)
            except Exception as e:
                print(f"EasyOCR 작업 중 오류 발생: {e}")
                result_q.put(f"[OCR Error: {e}]")
            finally:
                job_q.task_done()
        except queue.Empty:
            if not is_running:
                break

def on_key_press(event):
    """Handles key press events for the Tkinter window."""
    global is_running
    if event.keysym == 'Escape' or event.keysym == 'q':
        print("'q' 또는 'ESC' 키가 입력되어 프로그램을 종료합니다.")
        is_running = False

def main():
    cap = find_capture_device()
    if cap is None: sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"현재 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    print("EasyOCR 모델을 로드하는 중입니다...")
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    print("EasyOCR 모델 로드 완료.")
    device_info = reader.device
    print(f"EasyOCR 실행 장치: {device_info}")
    if "mps" in device_info or "cuda" in device_info:
        print("--> GPU 가속이 활성화되었습니다.")
    else:
        print("--> CPU를 사용하여 실행 중입니다.")

    job_queue = queue.Queue()
    result_queue = queue.Queue()
    ocr_thread = threading.Thread(target=ocr_worker, args=(job_queue, result_queue, reader), daemon=True)
    ocr_thread.start()

    root = tk.Tk()
    root.title("OCR 결과")
    root.geometry("600x400")
    text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 14))
    text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    # Bind the key press event to the root window
    root.bind("<KeyPress>", on_key_press)

    window_title = "OCR Application (Final)"
    cv2.namedWindow(window_title)

    status, is_flipping, mean_diff, last_capture_time, stabilizing_since = "Ready", False, 0.0, 0, None
    page_counter = 0
    previous_frame_gray = None
    USER_COOLDOWN, STABILIZATION_DELAY = 0.4, 0.5

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
                    text_area.insert(tk.END, ui_output)
                    text_area.see(tk.END)
                    output_file.write(ocr_text + "\n\n")
                    output_file.flush()
                    print(f"Page {page_counter} processed and saved.")
                    last_capture_time = time.time()
                    status = "Saved!"
                    result_queue.task_done()
                except queue.Empty:
                    pass

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                if previous_frame_gray is not None:
                    diff = cv2.absdiff(previous_frame_gray, gray)
                    mean_diff = np.mean(diff)

                    if mean_diff > 0.1 and not is_flipping and (time.time() - last_capture_time > USER_COOLDOWN):
                        status = "Flipping..."
                        is_flipping = True
                        stabilizing_since = None

                    if is_flipping:
                        if mean_diff == 0.0:
                            if stabilizing_since is None: stabilizing_since = time.time(); status = "Stabilizing..."
                        else: stabilizing_since = None; status = "Flipping..."
                        
                        if stabilizing_since is not None and (time.time() - stabilizing_since > STABILIZATION_DELAY):
                            job_queue.put(frame.copy()); status = "OCR Queued"; is_flipping = False; stabilizing_since = None

                    if not is_flipping and status not in ["Saved!", "OCR Queued"]: status = "Ready"
                    elif status == "Saved!" and (time.time() - last_capture_time > USER_COOLDOWN): status = "Ready"

                font = cv2.FONT_HERSHEY_SIMPLEX
                status_color = STATUS_COLORS.get(status, (0, 0, 255))
                cv2.putText(frame, f"Status: {status}", (10, 30), font, 0.8, status_color, 2, cv2.LINE_AA)
                cv2.imshow(window_title, frame)
                previous_frame_gray = gray.copy()

                root.update()
                root.update_idletasks()

                cv2.waitKey(1)
    finally:
        print("프로그램 종료 중...")
        job_queue.put(None)
        ocr_thread.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()
        try:
            root.destroy()
        except tk.TclError:
            pass # Window might already be closed
        print("모든 리소스를 해제하고 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()