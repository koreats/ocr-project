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
    """
    Applies a series of preprocessing steps to an image to improve OCR accuracy.
    """
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Deskewing
    # Threshold to get all non-black pixels
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Find coordinates of all non-zero pixels
    coords = np.column_stack(np.where(thresh > 0))

    # --- SAFETY CHECK: handle empty images ---
    if len(coords) == 0:
        # Return a default binarized image if there's no content to process
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # ----------------------------------------

    # Get the minimum area bounding rectangle
    angle = cv2.minAreaRect(coords)[-1]
    # The `cv2.minAreaRect` angle can be in [-90, 0). We need to correct it.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    # Rotate the image to deskew it
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 3. Adaptive Thresholding
    processed_image = cv2.adaptiveThreshold(
        rotated,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, # Block size
        2   # Constant subtracted from the mean
    )

    return processed_image

def ocr_worker(q):
    """
    Worker function that preprocesses and then OCRs a frame.
    """
    while True:
        frame = q.get()
        if frame is None:
            break

        try:
            # --- Preprocessing Step ---
            preprocessed_image = preprocess_for_ocr(frame)
            cv2.imshow("Preprocessed for OCR", preprocessed_image)
            # --------------------------

            text = pytesseract.image_to_string(preprocessed_image, lang='kor+eng')
            
            print("--- OCR Result ---")
            print(text.strip() if text.strip() else "[No text found]")
            print("--- End of Result ---\n")

        except Exception as e:
            print(f"OCR 작업 중 오류 발생: {e}")
        finally:
            q.task_done()

def main():
    cap = find_capture_device()
    if cap is None:
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"현재 해상도: {actual_width}x{actual_height}")

    ocr_queue = queue.Queue()
    ocr_thread = threading.Thread(target=ocr_worker, args=(ocr_queue,), daemon=True)
    ocr_thread.start()

    window_title = "Phase 6: Preprocessing OCR"
    cv2.namedWindow(window_title)
    cv2.namedWindow("Preprocessed for OCR") # Create window for the processed image

    is_flipping = False
    status = "Waiting..."
    previous_frame_gray = None
    mean_diff = 0.0
    stability_counter = 0
    STABILITY_THRESHOLD_FRAMES = 5

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

                if mean_diff > 1.0:
                    if not is_flipping:
                        is_flipping = True
                        status = "Flipping..."
                    stability_counter = 0
                else:
                    if is_flipping:
                        stability_counter += 1

                if is_flipping and stability_counter >= STABILITY_THRESHOLD_FRAMES:
                    ocr_queue.put(frame.copy())
                    status = "OCR Queued"
                    is_flipping = False
                    stability_counter = 0
                elif not is_flipping:
                    status = "Waiting..."

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
        print("프로그램 종료 중...")
        ocr_queue.put(None)
        ocr_queue.join()
        ocr_thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        print("모든 리소스를 해제하고 프로그램을 종료합니다.")

if __name__ == "__main__":
    main()
