import cv2
import sys
import numpy as np

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
    Main function to capture, process, and display video feed with change detection.
    """
    cap = find_capture_device()

    if cap is None:
        print("오류: 캡처 장치를 찾을 수 없습니다. 인덱스 0, 1, 2를 확인했습니다.", file=sys.stderr)
        sys.exit(1)

    window_title = "Phase 2: Visual Debug Feed"
    cv2.namedWindow(window_title)

    previous_frame_gray = None
    mean_diff = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("오류: 비디오 스트림에서 프레임을 읽을 수 없습니다.", file=sys.stderr)
                break

            # 현재 프레임을 그레이스케일로 변환
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            status = "Waiting..."

            # 첫 프레임이 아닐 경우에만 차이 계산
            if previous_frame_gray is not None:
                # 현재 프레임과 이전 프레임 간의 평균 절대 차이 계산
                diff = cv2.absdiff(previous_frame_gray, current_frame_gray)
                mean_diff = np.mean(diff)

                # 임계값(25.0)을 기준으로 상태 결정
                if mean_diff > 25.0:
                    status = "CHANGE DETECTED!"
            
            # 시각적 디버깅 정보를 프레임에 추가
            font = cv2.FONT_HERSHEY_SIMPLEX
            status_text = f"Status: {status}"
            diff_text = f"Difference: {mean_diff:.2f}"

            cv2.putText(frame, status_text, (10, 30), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, diff_text, (10, 70), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # 결과 프레임을 화면에 표시
            cv2.imshow(window_title, frame)

            # 다음 루프를 위해 현재 프레임을 이전 프레임으로 업데이트
            previous_frame_gray = current_frame_gray.copy()

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