import cv2 as cv
import numpy as np
import ast

# Read images
original_img = cv.imread('Backend_Engineer/original_image.png')

with open('Backend_Engineer/landmarks.txt', 'r') as f:
    landmarks_data = ast.literal_eval(f.read())

landmarks = landmarks_data['landmarks'][0]
points = np.array([[int(lm['x']), int(lm['y'])] for lm in landmarks], dtype=np.int32)

debug_img = original_img.copy()
for i in range(len(points)):
    point = points[i]
    cv.circle(debug_img, tuple(point), 3, (0, 255, 0), -1)
    cv.putText(debug_img, str(i), tuple(point + [5, 5]), 
               cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

cv.imshow('face Landmarks', debug_img)
cv.imwrite('Backend_Engineer/face_landmarks.png', debug_img)

cv.waitKey(0)
cv.destroyAllWindows()