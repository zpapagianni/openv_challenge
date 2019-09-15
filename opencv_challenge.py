import numpy as np
import cv2

cap = cv2.VideoCapture("Motion_cut.mp4")
#cap = cv2.VideoCapture("NoMotion.mp4")

ret, frame1 = cap.read()
ret, frame2 = cap.read()

print(frame1.shape)
count=0
while cap.isOpened():

    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3), 0)
    median = cv2.medianBlur(blur, 3)
    _, thresh = cv2.threshold(median, 30, 255, cv2.THRESH_BINARY)

    kernel=np.ones((15,15), np.uint8)
    dilated = cv2.dilate(thresh,kernel,iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 20000:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite("./Motions/frame%d.jpg" %count, frame1)
        count += 1

    image = cv2.resize(frame1, (960, 540))
    cv2.imshow("feed", image)

    frame1 = frame2
    ret, frame2 = cap.read()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

