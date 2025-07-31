import cv2
import numpy as np

fish = cv2.imread('../img/fish.jpg')
# cv2.imshow('fish', fish)

rows, cols = fish.shape[0:2]
dx = 100
dy = 50

matrix = np.float32([[1, 0, dx],
                     [0, 1, dy]])

m_small = np.float32([[0.5, 0, 0],
                      [0, 0.5, 0]])

m_big = np.float32([[2, 0, 0],
                    [0, 2, 0]])

m45 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
m90 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1.5)

# 이미지 이동
move_fish_1 = cv2.warpAffine(fish, matrix, (cols + dx, rows + dy))

move_fish_2 = cv2.warpAffine(fish, matrix, (cols + dx, rows + dy), None, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT,(255, 0, 0))

move_fish_3 = cv2.warpAffine(fish, matrix, (cols + dx, rows + dy), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)

# 이미지 축소 & 확대
small_fish = cv2.warpAffine(fish, m_small, (int(cols * 0.5), int(rows * 0.5)))
big_fish = cv2.warpAffine(fish, m_big, (int(cols * 2), int(rows * 2)))

# 이미지 회전
rotation_fish_45 = cv2.warpAffine(fish, m45, (cols, rows))
rotation_fish_90 = cv2.warpAffine(fish, m90, (cols, rows))

cv2.imshow('move_fish_1', move_fish_1)
cv2.imshow('move_fish_2', move_fish_2)
cv2.imshow('move_fish_3', move_fish_3)
cv2.imshow('small_fish', small_fish)
cv2.imshow('rotation_fish_45', rotation_fish_45)
cv2.imshow('rotation_fish_90', rotation_fish_90)
cv2.waitKey(0)
cv2.destroyAllWindows()
