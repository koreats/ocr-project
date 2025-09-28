import cv2
import sys
import numpy as np
import os
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

def main():
    """
    Main function to perform OCR in real-time when the screen stabilizes.
    """
    cap = find_capture_device()

    if cap is None:
        print("오류: 캡처 장치를 찾을 수 없습니다. 인덱스 0, 1, 2를 확인했습니다.", file=sys.stderr)
        sys.exit(1)

    # --- 해상도 설정 (1080p로 유지) ---
    requested_width = 1920
    requested_height = 1080
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("------------------------------------")
    print(f"현재 해상도: {actual_width}x{actual_height}")
    print("------------------------------------")
    # --------------------------------

    window_title = "Phase 4: OCR Integration"
    cv2.namedWindow(window_title)

    is_flipping = False
    status = "Waiting..."

    previous_frame_gray = None
    mean_diff = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("오류: 비디오 스트림에서 프레임을 읽을 수 없습니다.", file=sys.stderr)
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if previous_frame_gray is not None:
                diff = cv2.absdiff(previous_frame_gray, gray)
                mean_diff = np.mean(diff)

                # --- State machine with OCR logic ---
                if mean_diff > 1.0 and not is_flipping:
                    is_flipping = True
                    status = "Flipping..."
                
                elif is_flipping and mean_diff == 0.0:
                    # Convert frame for Pytesseract (OpenCV BGR -> PIL RGB)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)

                    # Perform OCR
                    try:
                        text = pytesseract.image_to_string(pil_image, lang='kor+eng')
                        print("---" + " OCR Result " + "---")
                        print(text.strip() if text else "[No text found]")
                        print("---" + " End of Result " + "---" + "\n")
                        status = "OCR Complete!"
                    except pytesseract.TesseractNotFoundError:
                        print("오류: Tesseract-OCR 엔진을 찾을 수 없습니다.")
                        print("시스템에 Tesseract를 설치해야 합니다. (예: brew install tesseract)")
                        # Stop the loop if Tesseract is not found
                        return 

                    is_flipping = False

                elif is_flipping:
                    status = "Flipping..."

                else:
                    status = "Waiting..."
                # ----------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX
            status_text = f"Status: {status}"
            diff_text = f"Difference: {mean_diff:.2f}"

            cv2.putText(frame, status_text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, diff_text, (10, 70), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_title, frame)

            previous_frame_gray = gray.copy()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' 키가 입력되어 프로그램을 종료합니다.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("캡처를 종료하고 모든 창을 닫았습니다.")

if __name__ == "__main__":
    main()