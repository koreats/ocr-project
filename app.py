import cv2
import sys
import numpy as np
import threading
import queue
import easyocr
import time

def find_capture_device():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
    return None

def ocr_worker(job_q, result_q, reader):
    while True:
        frame = job_q.get()
        if frame is None: break
        try:
            result = reader.readtext(frame)
            text = "\n".join([item[1] for item in result])
            result_q.put(text)
        except Exception as e:
            print(f"EasyOCR 작업 중 오류 발생: {e}")
            result_q.put(f"[OCR Error: {e}]")
        finally:
            job_q.task_done()

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

    window_title = "OCR Application (Final)"
    cv2.namedWindow(window_title)

    # --- App Variables & UI ---
    status = "Ready"
    is_flipping = False
    previous_frame_gray = None
    mean_diff = 0.0
    last_capture_time = 0
    stabilizing_since = None
    COOLDOWN_SECONDS = 1.5
    STABILIZATION_DELAY = 0.5

    STATUS_COLORS = {
        "Ready": (0, 255, 0),          # Green
        "Flipping...": (0, 255, 255),   # Yellow
        "Stabilizing...": (255, 255, 0),# Cyan
        "OCR Queued": (255, 0, 0),      # Blue
        "Saved!": (255, 0, 255)         # Magenta
    }
    # -------------------------

    try:
        with open("output.txt", "a", encoding="utf-8") as output_file:
            while True:
                ret, frame = cap.read()
                if not ret: break

                try:
                    ocr_text = result_queue.get_nowait()
                    output_file.write(ocr_text + "\n\n")
                    output_file.flush()
                    print("OCR result saved to output.txt")
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

                    # --- Final State Machine ---
                    if mean_diff > 0.1 and not is_flipping and (time.time() - last_capture_time > COOLDOWN_SECONDS):
                        status = "Flipping..."
                        is_flipping = True
                        stabilizing_since = None

                    if is_flipping:
                        if mean_diff == 0.0:
                            if stabilizing_since is None:
                                stabilizing_since = time.time()
                                status = "Stabilizing..."
                        else:
                            stabilizing_since = None
                            status = "Flipping..."
                        
                        if stabilizing_since is not None and (time.time() - stabilizing_since > STABILIZATION_DELAY):
                            job_queue.put(frame.copy())
                            status = "OCR Queued"
                            is_flipping = False
                            stabilizing_since = None

                    if not is_flipping and status not in ["Saved!", "OCR Queued"]:
                        status = "Ready"
                    elif status == "Saved!" and (time.time() - last_capture_time > COOLDOWN_SECONDS):
                        status = "Ready"
                    # ---------------------------

                font = cv2.FONT_HERSHEY_SIMPLEX
                status_color = STATUS_COLORS.get(status, (0, 0, 255)) # Default to Red if status is unknown
                cv2.putText(frame, f"Status: {status}", (10, 30), font, 0.8, status_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Difference: {mean_diff:.2f}", (10, 70), font, 0.8, (255,255,255), 2, cv2.LINE_AA)

                cv2.imshow(window_title, frame)
                previous_frame_gray = gray.copy()

                if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print("프로그램 종료 중... OCR 스레드에 종료 신호를 보냅니다.")
        job_queue.put(None)      # Signal the worker to stop
        ocr_thread.join(timeout=5) # Wait up to 5 seconds for the thread to finish
        
        if ocr_thread.is_alive():
            print("경고: OCR 스레드가 시간 내에 종료되지 않았습니다.")

        cap.release()
        cv2.destroyAllWindows()
        print("모든 리소스를 해제하고 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
