import cv2
import os

folder= "./Frame_Folder"
width= 1920
height= 1080

def get_frame_number(filename):
    return int(filename.replace('frame', '').replace('.jpg', ''))

cap = cv2.VideoCapture('./Social_Network.avi')
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        print("End of video or failed to read the frame.")
        break
   
    cv2.imwrite("./Frame_Folder/frame%d.jpg" % count, frame)
    count = count + 1
    

cap.release() 

# Read Files
files = [f for f in os.listdir(folder)]
files.sort(key=get_frame_number)

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('Social_Net_diff.avi', fourcc, 24, (width, height),False)

for i in range(len(files) - 1):
    img1 = cv2.imread(os.path.join(folder, files[i]), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(folder, files[i + 1]), cv2.IMREAD_GRAYSCALE)

    diff = cv2.absdiff(img2, img1)
    cv2.imwrite("./Diff_Folder/Diff%d.jpg" % i, diff)
    video.write(diff)
