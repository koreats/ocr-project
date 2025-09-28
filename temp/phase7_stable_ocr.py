import cv2
import sys
import numpy as np
import threading
import queue
import pytesseract
from PIL import Image

def find_capture_device():
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
    return None

def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10: # Not enough points to deskew
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    processed_image = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return processed_image

def ocr_worker(job_q, result_q):
    """
    Worker thread: gets a frame, preprocesses, OCRs, and puts result back.
    """
    while True:
        frame = job_q.get()
        if frame is None:
            break
        try:
            preprocessed_image = preprocess_for_ocr(frame)
            text = pytesseract.image_to_string(preprocessed_image, lang='kor+eng')
            result_q.put((preprocessed_image, text))
        except Exception as e:
            print(f"OCR 작업 중 오류 발생: {e}")
            result_q.put((None, "")) # Put an empty result to avoid blocking
        finally:
            job_q.task_done()

def main():
    cap = find_capture_device()
    if cap is None: sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    print(f"현재 해상도: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    job_queue = queue.Queue()
    result_queue = queue.Queue()
    ocr_thread = threading.Thread(target=ocr_worker, args=(job_queue, result_queue), daemon=True)
    ocr_thread.start()

    window_title = "Phase 7: Stable OCR"
    cv2.namedWindow(window_title)
    cv2.namedWindow("Preprocessed Image")

    is_flipping, status, previous_frame_gray, mean_diff = False, "Waiting...", None, 0.0
    stability_counter, STABILITY_THRESHOLD_FRAMES = 0, 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # --- Check for and display OCR results from the worker ---
            try:
                processed_img, ocr_text = result_queue.get_nowait()
                if processed_img is not None:
                    cv2.imshow("Preprocessed Image", processed_img)
                    print("---")
                    print(ocr_text.strip() if ocr_text.strip() else "[No text found]")
                    print("---")
                    result_queue.task_done()
            except queue.Empty:
                pass # No new result
            # ---------------------------------------------------------

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if previous_frame_gray is not None:
                diff = cv2.absdiff(previous_frame_gray, gray)
                mean_diff = np.mean(diff)

                if mean_diff > 1.0:
                    if not is_flipping: status = "Flipping..."
                    is_flipping = True
                    stability_counter = 0
                else:
                    if is_flipping: stability_counter += 1

                if is_flipping and stability_counter >= STABILITY_THRESHOLD_FRAMES:
                    job_queue.put(frame.copy())
                    status = "OCR Queued"
                    is_flipping = False
                    stability_counter = 0
                elif not is_flipping: status = "Waiting..."

            # --- Update UI in main thread ---
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"Status: {status}", (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Difference: {mean_diff:.2f}", (10, 70), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Stability: {stability_counter}/{STABILITY_THRESHOLD_FRAMES}", (10, 110), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_title, frame)
            # --------------------------------

            previous_frame_gray = gray.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        print("프로그램 종료 중...")
        job_queue.put(None) # Signal worker to exit
        job_queue.join()
        result_queue.join()
        ocr_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        print("모든 리소스를 해제하고 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
