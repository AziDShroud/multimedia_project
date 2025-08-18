import cv2
cap = cv2.VideoCapture('./Social_Network.avi')
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("End of video or failed to read the frame.")
        break
    #  cv2.imshow('window-name', frame)
    cv2.imwrite("./Frame_Folder/frame%d.jpg" % count, frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # destroy all opened windows    