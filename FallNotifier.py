from __future__ import print_function
import cv2
from pprint import pprint
from ultralytics import YOLO
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv
import os


def get_label(num):
    if num == 0:
        return 'NOT FALL'
    if num == 1:
        return 'FALL'
    else:
        return -1


#annotate video for eacj frame: 0 = not fall, 1 = fall, creating an image file for each in jpg format
def annotateVideo(video):
    cnt = 0
    columns = ['images', 'labels']
    data = []
    csvFile = open('annotation.csv', 'w')
    csvwriter = csv.writer(csvFile)
    csvwriter.writerow(columns)
    path = "/Users/dorian/documents/imagerecognition/FallAssistant/images"

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        cv2.imshow("Video",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #ask for input
        cnt += 1
        if cnt % 10 == 0:
            result = input("Is this a fall? (y/n)")
            if result == 'y':
                result = 1
            else:
               result = 0
            
            #initialize new csv file for writing
            imageName = 'trainImage' + str(cnt) + '.jpg'
            #write a file with 2 columns:  image name and result
            csv.write(imageName + ',' + str(result))
            #download image to local directory
            cv2.imwrite(imageName, frame)
            print(imageName + ',' + str(result))
            finalpath = os.path.join(path, imageName)
            cv2.imwrite(finalpath, frame)
            print(imageName)
            data.append([imageName, result])


    csvwriter.writerows(data)

#TwilioNotifier.py
from json_minify import json_minify
import json
from imutils.io import TempFile

class Conf:
	def __init__(self, confPath):
		# load and store the configuration and update the object's dictionary
		conf = json.loads(json_minify(open(confPath).read()))
		self.__dict__.update(conf)
        
	def __getitem__(self, k):
		# return the value associated with the supplied key
		return self.__dict__.get(k, None)
    

from twilio.rest import Client
import boto3
from threading import Thread

class TwilioNotifier:
    def __init__(self, conf):
        # store the configuration object
        self.conf = conf
    
    def send(self, msg, tempVideo):

        # start a thread to upload the file and send it
        #t = Thread(target=self._send, args=(msg, tempVideo,))
        #t.start()

        # create a s3 client object
        s3 = boto3.client("s3", aws_access_key_id=self.conf["aws_access_key_id"], aws_secret_access_key=self.conf["aws_secret_access_key"])
        
        # get the filename and upload the video in public read mode
        filename = "outToUpload.mp4"
        s3.upload_file("/Users/dorian/documents/imagerecognition/FallAssistant/outToUpload.mp4", self.conf["s3_bucket"], filename, ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"})
        location = s3.get_bucket_location(Bucket=self.conf["s3_bucket"])["LocationConstraint"]
       
        url = "https://falldetectionstorage.s3.amazonaws.com/outToUpload.mp4"
        
        # initialize the twilio client and send the message
        client = Client(self.conf["twilio_sid"], self.conf["twilio_auth"])
        client.messages.create(to=self.conf["twilio_to"], from_=self.conf["twilio_from"], body=msg, media_url = [url])
                

    

conf = Conf("/Users/dorian/documents/imagerecognition/FallAssistant/configuration/conf.json")


def checkIfToSend(prev, curr,W,H, writer, tempVideo):
    tn = TwilioNotifier(conf)
    if prev == 'NOT FALL' and curr == 'FALL':
        tempVideo = TempFile(ext=".mp4")
        print(tempVideo.path)
        writer = cv2.VideoWriter("/Users/dorian/documents/imagerecognition/FallAssistant/outToUpload.mp4", cv2.VideoWriter_fourcc(*'MP4V'), 30, (W, H))
        print("New FALL DETECTED")
    elif prev == 'FALL' and curr == 'NOT FALL':
        writer.release()
        writer = None
        print("FALL ENDED")
        msg = "FALL DETECTED"
        tn.send(msg, tempVideo)
        return ["message sent", tempVideo]
    elif prev == 'FALL' and curr == 'FALL':
        print("FALL CONTINUES")
    elif prev == 'NOT FALL' and curr == 'NOT FALL':
        print("NO FALL")
    
    return [writer, tempVideo]

def main():
    video = cv2.VideoCapture('/Users/dorian/Downloads/fallingForTraining.mp4')
    cnt = 0
    model = tf.keras.models.load_model('/Users/dorian/documents/imagerecognition/FallAssistant/combinedmodel-2.h5')
    result =""
    print(model.summary())
    prev = None
    writer = None
    tempVideo = None
    maxMessages = 5
    while video.isOpened():
        ret, frame = video.read()
        copy = frame
        if not ret:
            break
        cnt += 1
        frame = cv2.resize(frame, (96,96))
        frame = np.expand_dims(frame, axis=0)
        if cnt % 5 == 0:
            
            predicted_labels = (model.predict(frame) >= 0.5).astype('int64').flatten()
            result = get_label(predicted_labels[0])
            #put text in the middle of the screen
            [writer, tempVideo] = checkIfToSend(prev, result, int(video.get(3)), int(video.get(4)), writer, tempVideo)
            if writer == "message sent":
                writer = None
                print("message sent")
                maxMessages -= 1
                break
                if maxMessages == 0:
                    break
            prev = result
        #change frame back to               
        frame = copy
        if writer != None :
            writer.write(frame)
        cv2.putText(frame, result, (int(frame.shape[1]/2),int(frame.shape[0]/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    video.release()
    if writer != None:
        writer.release()
    cv2.destroyAllWindows()


main()
