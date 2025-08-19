import cv2
import os
folder= "./Frame_Folder"
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
files = [f for f in os.listdir(folder)]

def get_frame_number(filename):
    return int(filename.replace('frame', '').replace('.jpg', ''))

# Sort files
files.sort(key=get_frame_number)

for i in range(len(files) - 1):
    img1 = cv2.imread(os.path.join(folder, files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(folder, files[i + 1]), cv2.IMREAD_GRAYSCALE)

    diff = cv2.absdiff(img2, img1)
    cv2.imwrite("./Diff_Folder/Diff%d.jpg" % i, diff)
    