import cv2
import csv
import os
def annotateVideo(videoPath):

    cnt = 0
    columns = ['images', 'labels']
    data = []
    csvFile = open('annotation.csv', 'w')
    csvwriter = csv.writer(csvFile)
    csvwriter.writerow(columns)
    path = "/Users/dorian/documents/imagerecognition/images"
    video = cv2.VideoCapture(videoPath)
    binary = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow("Video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #detect if space bar is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if binary == 0:
                binary = 1
            else:
                binary = 0

       
        #ask for input
        cnt += 1
        if cnt % 10 == 0:
            #initialize new csv file for writing
            imageName = 'trainImage' + str(cnt) + '.jpg'
            #write a file with 2 columns:  image name and result
            #csv.write(imageName + ',' + str(binary))
            #download image to local directory
            #cv2.imwrite(imageName, frame)
            print(imageName + ',' + str(binary))
            finalpath = os.path.join(path, imageName)
            cv2.imwrite(finalpath, frame)
            #print(imageName)
            data.append([imageName, binary])
        


    csvwriter.writerows(data)

annotateVideo("/Users/dorian/documents/imagerecognition/fallingForTraining.mp4")