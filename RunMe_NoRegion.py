import cv2, imutils
from CrowdEstimation import Detector

cap = cv2.VideoCapture('human.avi')

while True:
    ret, frame = cap.read()
    # frame = imutils.resize(frame, width=800)
    frame = Detector(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
