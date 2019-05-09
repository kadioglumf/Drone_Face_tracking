from __future__ import division
import cv2
import time
import sys

class FaceDetect(object):
    def __init__(self):
        DNN = "TF"
        if DNN == "CAFFE":
            modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
            configFile = "models/deploy.prototxt"
            self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        else:
            modelFile = "models/opencv_face_detector_uint8.pb"
            configFile = "models/opencv_face_detector.pbtxt"
            self.net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    
    def detectFaceOpenCVDnn(self,frame):
        
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)

        self.net.setInput(blob)
        detections = self.net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.8:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                text = "{:.2f}%".format(confidence * 100)
                y = y1 - 10 if y1 - 10> 10 else y1 + 10
                
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
                cv2.putText(frameOpencvDnn,text,(x1,y),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0, 255, 0),2)
        return frameOpencvDnn, bboxes




