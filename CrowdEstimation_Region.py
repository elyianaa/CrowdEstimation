import cv2

# Variable Mouse
drawing = False
point1 = ()
point2 = ()
Mouse_count = False


def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing
    global Mouse_count

    # Mouse
    if not Mouse_count:
        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing is False:
                drawing = True
                point1 = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing is True:
                point2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            Mouse_count = True


# create VideoCapture object and read from video file
cap = cv2.VideoCapture('human.avi')

cv2.namedWindow("Crowd Estimation")
cv2.setMouseCallback("Crowd Estimation", mouse_drawing)

while True:
    ret, frame = cap.read()
    # pretrained models
    persons_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    if point1 and point2:

        # Rectangle marker
        r = cv2.rectangle(frame, point1, point2, (0, 250, 255), 3)
        frame_ROI = frame[point1[1]:point2[1], point1[0]:point2[0]]

        # Detect People
        if drawing is False:
            try:
                # convert video into gray scale of each frames
                ROI_grayscale = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)

                # detect person_ROI in the video
                person_ROI = persons_cascade.detectMultiScale(ROI_grayscale, 1.1, 1)
                for (x, y, w, h) in person_ROI:
                    cv2.rectangle(frame_ROI, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame_ROI, "People: " + str(person_ROI.shape[0]), (10, frame_ROI.shape[0] - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            except:
                print("Ignore An exception occurred")

    cv2.imshow("Crowd Estimation", frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
