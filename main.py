import cv2
import mediapipe as mp

# Mediapipe 손 인식 모듈 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 화면을 좌우 반전(미러 모드)
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Mediapipe는 RGB 이미지를 사용
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    # 손 인식이 되었을 경우
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((cx, cy))

            # 손가락의 펴짐 상태를 저장할 리스트 (엄지, 검지, 중지, 약지, 새끼)
            finger_status = []

            # 엄지: 일반적으로 x 좌표를 비교 (오른손 기준)
            if lm_list[4][0] < lm_list[3][0]:
                finger_status.append(1)
            else:
                finger_status.append(0)

            # 검지: tip (8)과 pip (6) 비교 (y 좌표가 작으면 위쪽에 있음)
            if lm_list[8][1] < lm_list[6][1]:
                finger_status.append(1)
            else:
                finger_status.append(0)

            # 중지: tip (12)과 pip (10)
            if lm_list[12][1] < lm_list[10][1]:
                finger_status.append(1)
            else:
                finger_status.append(0)

            # 약지: tip (16)과 pip (14)
            if lm_list[16][1] < lm_list[14][1]:
                finger_status.append(1)
            else:
                finger_status.append(0)

            # 새끼: tip (20)과 pip (18)
            if lm_list[20][1] < lm_list[18][1]:
                finger_status.append(1)
            else:
                finger_status.append(0)

            total_fingers = sum(finger_status)

            # 욕 제스처 판 단: 오직 중지만 펴진 경우
            # finger_status의 인덱스: [엄지, 검지, 중지, 약지, 새끼]
            if finger_status[2] == 1 and finger_status[0] == 0 and finger_status[1] == 0 and finger_status[3] == 0 and finger_status[4] == 0:
                display_text = "fuck"  # 욕 이모티콘
            else:
                display_text = str(total_fingers)  # 펴진 손가락 개수를 숫자로 표시

            # 화면에 텍스트 표시
            cv2.putText(frame, display_text, (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 5)

            # 손의 랜드마크와 연결선 그리기
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Webcam", frame)
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
