import cv2
import sys

def find_capture_device():
    """
    Tries to find a connected video capture device by checking indices 0, 1, and 2.
    This is to locate a device like an Elgato capture card which may not be the default.
    """
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"캡처 장치를 인덱스 {i}에서 찾았습니다.")
            return cap
    return None

def main():
    """
    Main function to capture and display video feed.
    """
    cap = find_capture_device()

    if cap is None:
        print("오류: 캡처 장치를 찾을 수 없습니다. 인덱스 0, 1, 2를 확인했습니다.", file=sys.stderr)
        sys.exit(1)

    window_title = "Phase 1: Live Capture Feed"
    cv2.namedWindow(window_title)

    try:
        while True:
            # 프레임 단위로 캡처
            ret, frame = cap.read()
            if not ret:
                print("오류: 비디오 스트림에서 프레임을 읽을 수 없습니다.", file=sys.stderr)
                break

            # 캡처된 프레임을 화면에 표시
            cv2.imshow(window_title, frame)

            # 'q' 키를 누르면 루프를 빠져나감
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' 키가 입력되어 프로그램을 종료합니다.")
                break
    finally:
        # 모든 작업이 끝나면 캡처 객체를 해제
        cap.release()
        # 모든 OpenCV 창을 닫음
        cv2.destroyAllWindows()
        print("캡처를 종료하고 모든 창을 닫았습니다.")

if __name__ == "__main__":
    main()
