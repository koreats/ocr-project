import cv2
import sys
import numpy as np
import threading
import queue
import pytesseract
from PIL import Image

def find_capture_device():
    """
    Tries to find a connected video capture device by checking indices 0, 1, and 2.
    """
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
    return None

def ocr_worker(q):
    """
    Worker function that runs in a separate thread to perform OCR.
    """
    while True:
        # Get a frame from the queue
        frame = q.get()
        if frame is None:  # Sentinel value to exit
            break

        try:
            # Convert frame for Pytesseract (OpenCV BGR -> PIL RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            # Perform OCR
            text = pytesseract.image_to_string(pil_image, lang='kor+eng')
            
            print("--- OCR Result ---")
            print(text.strip() if text.strip() else "[No text found]")
            print("--- End of Result ---\
")

        except pytesseract.TesseractNotFoundError:
            print("오류: Tesseract-OCR 엔진을 찾을 수 없습니다.")
            print("시스템에 Tesseract를 설치해야 합니다. (예: brew install tesseract)")
            # To prevent repeated errors, we can consider a way to stop the main loop
            # For now, this will print once per frame in the queue.
        except Exception as e:
            print(f"OCR 작업 중 오류 발생: {e}")
        finally:
            q.task_done()

def main():
    """
    Main function with a separate thread for OCR to ensure smooth video.
    """
    cap = find_capture_device()
    if cap is None:
        print("오류: 캡처 장치를 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)

    # --- Resolution and FPS setup (using 1080p) ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"현재 해상도: {actual_width}x{actual_height}")
    # -------------------------------------------

    # --- Threading and Queue Setup ---
    ocr_queue = queue.Queue()
    ocr_thread = threading.Thread(target=ocr_worker, args=(ocr_queue,), daemon=True)
    ocr_thread.start()
    # ---------------------------------

    window_title = "Phase 5: Multithreaded OCR"
    cv2.namedWindow(window_title)

    is_flipping = False
    status = "Waiting..."
    previous_frame_gray = None
    mean_diff = 0.0

    # --- Stability Counter Setup ---
    stability_counter = 0
    STABILITY_THRESHOLD_FRAMES = 5  # Must be stable for 5 frames
    # -------------------------------

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if previous_frame_gray is not None:
                diff = cv2.absdiff(previous_frame_gray, gray)
                mean_diff = np.mean(diff)

                # --- State machine with stability counter ---
                if mean_diff > 1.0:
                    if not is_flipping:
                        is_flipping = True
                        status = "Flipping..."
                    stability_counter = 0 # Reset counter on any motion
                else: # mean_diff <= 1.0
                    if is_flipping:
                        stability_counter += 1

                if is_flipping and stability_counter >= STABILITY_THRESHOLD_FRAMES:
                    ocr_queue.put(frame.copy())
                    status = "OCR Queued"
                    is_flipping = False
                    stability_counter = 0
                elif not is_flipping:
                    status = "Waiting..."
                # -------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX
            status_text = f"Status: {status}"
            diff_text = f"Difference: {mean_diff:.2f}"
            stability_text = f"Stability: {stability_counter}/{STABILITY_THRESHOLD_FRAMES}"

            cv2.putText(frame, status_text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, diff_text, (10, 70), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, stability_text, (10, 110), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_title, frame)
            previous_frame_gray = gray.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("프로그램 종료 중... OCR 작업이 완료될 때까지 잠시 기다려 주세요.")
        # Signal the OCR worker to exit and wait for it
        ocr_queue.put(None)
        ocr_queue.join()
        ocr_thread.join(timeout=2) # Wait max 2s for thread to finish
        
        cap.release()
        cv2.destroyAllWindows()
        print("모든 리소스를 해제하고 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
