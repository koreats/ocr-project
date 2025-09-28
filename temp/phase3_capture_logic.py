import cv2
import sys
import numpy as np
import os

def find_capture_device():
    """
    Tries to find a connected video capture device by checking indices 0, 1, and 2.
    """
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
    return None

def main():
    """
    Main function to capture images using user-defined precise thresholds.
    """
    cap = find_capture_device()

    if cap is None:
        print("오류: 캡처 장치를 찾을 수 없습니다. 인덱스 0, 1, 2를 확인했습니다.", file=sys.stderr)
        sys.exit(1)

    CAPTURE_DIR = "captures"
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    print(f"이미지 저장 폴더: '{CAPTURE_DIR}'")

    # --- 현재 해상도 확인 ---
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("------------------------------------")
    print(f"현재 캡처 해상도: {width} x {height}")
    print("------------------------------------")
    # -------------------------

    window_title = "Phase 3: Capture Logic (User-Defined)"
    cv2.namedWindow(window_title)

    is_flipping = False
    capture_count = 0
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

                # --- State machine with user-defined logic ---
                if mean_diff > 1.0 and not is_flipping:
                    is_flipping = True
                    status = "Flipping..."
                
                elif is_flipping and mean_diff == 0.0:
                    capture_count += 1
                    filename = os.path.join(CAPTURE_DIR, f"capture_{capture_count:03d}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved {filename}")
                    status = "Capture Done!"
                    is_flipping = False

                elif is_flipping:
                    status = "Flipping..."

                else:
                    status = "Waiting..."
                # ----------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX
            status_text = f"Status: {status}"
            diff_text = f"Difference: {mean_diff:.2f}"
            flipping_text = f"is_flipping: {is_flipping}"

            cv2.putText(frame, status_text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, diff_text, (10, 70), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, flipping_text, (10, 110), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

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
