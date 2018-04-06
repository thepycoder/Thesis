import cv2

cap = cv2.VideoCapture()
vid = cap.open("../Footage/TestSeq1.mp4")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    print(ret)

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()