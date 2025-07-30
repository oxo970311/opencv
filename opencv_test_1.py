# 검은 선을 따라 주행하다가 장애물을 감지하면 멈추는 시스템
# 장애물(노란색 사각형)영역에 검은색이 아닌 다른값이 감지되면 obstacle!
# 장애물 테스트는 주먹을 쥐어 장애물(노란색 사각형)영역 을 가리는 식으로 테스트 하였음

# 핵심 내용
# ROI 를 활용해 관심영역 지정
# cv2.inRange 함수를 이용해 색깔 범위 지정
# cv2.cv2.GaussianBlur 함수를 이용해 이미지 블러 적용
# contour 를 활용해 이미지 윤곽선 추출
# while 반복문을 통해 주행 및 장애물 탐지

import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ROI_Y_START = int(480 * 2 / 3)
ROI_Y_END = 480
ROI_X_START = 0
ROI_X_END = 640

OBS_Y_START = 320
OBS_Y_END = 480
OBS_X_START = 280
OBS_X_END = 360

lower_black = (0, 0, 0)
upper_black = (60, 60, 60)

lower_rainbow = (20, 20, 20)
upper_rainbow = (235, 235, 235)

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:

            # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img, (5, 5), 0)
            blm = cv2.inRange(img_blur, lower_black, upper_black)
            rainbow = cv2.inRange(img_blur, lower_rainbow, upper_rainbow)
            edges = cv2.Canny(blm, 50, 150)


            cv2.inRange(img_blur, lower_rainbow, upper_rainbow)

            # 관심 영역(ROI) 설정
            roi = edges[ROI_Y_START:ROI_Y_END, ROI_X_START:ROI_X_END]
            rect = cv2.rectangle(img, (ROI_X_START, ROI_Y_START), (ROI_X_END, ROI_Y_END), (0, 0, 255), 1)
            obs_rect = cv2.rectangle(img, (OBS_X_START, OBS_Y_START), (OBS_X_END, OBS_Y_END), (255, 0, 0), 2)
            rainbow_obs_mask = rainbow[OBS_Y_START:OBS_Y_END, OBS_X_START:OBS_X_END]
            contours, _ = cv2.findContours(rainbow_obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            is_obstacle_detected = False
            min_obstacle_area = 500  # 장애물로 간주할 최소 면적 (픽셀). 이 값은 튜닝이 필수입니다!

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_obstacle_area:
                    is_obstacle_detected = True

                    # (옵션) 감지된 장애물을 원본 img에 사각형으로 표시
                    x, y, w, h = cv2.boundingRect(cnt)
                    x_orig_obs = x + OBS_X_START
                    y_orig_obs = y + OBS_Y_START
                    cv2.rectangle(img, (x_orig_obs, y_orig_obs), (x_orig_obs + w, y_orig_obs + h), (0, 255, 255),
                                  3)  # 노란색 사각형
                    print('obstacle!')
                    break  # 첫 번째 감지된 유효 장애물만으로 판단한다면

            # 허프 변환으로 선 감지
            # minLineLength: 선의 최소 길이, maxLineGap: 선 사이의 최대 허용 간격
            lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    y1_orig = y1 + ROI_Y_START
                    y2_orig = y2 + ROI_Y_START
                    cv2.line(img, (x1, y1_orig), (x2, y2_orig), (0, 255, 0), 2)

                    print('run..')

            cv2.imshow('camera', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            print("no frame")
            break
else:
    print("can't open camera")

cap.release()
cv2.destroyAllWindows()