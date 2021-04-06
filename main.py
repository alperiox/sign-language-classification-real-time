import cv2
import numpy as np
import os

labels = {0: 'a',
 1: 'b',
 2: 'c',
 3: 'd',
 4: 'e',
 5: 'f',
 6: 'g',
 7: 'h',
 8: 'i',
 9: 'j',
 10: 'k',
 11: 'l',
 12: 'm',
 13: 'n',
 14: 'o',
 15: 'p',
 16: 'q',
 17: 'r',
 18: 's',
 19: 't',
 20: 'u',
 21: 'v',
 22: 'w',
 23: 'x',
 24: 'y',
 25: 'z'}


def preprocess_frame(frame):
    frame = cv2.resize(frame, (28, 28))
    frame = frame / 255.
    return frame

net = cv2.dnn.readNetFromTensorflow('frozen_graph.pb')

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    
    frame = cv2.rectangle(frame, (400,25), (588, 208), (0, 255, 0))
    
    input_frame = frame[25:208, 400:588]
    preprocessed_frame = cv2.cvtColor(input_frame, cv2.COLOR_RGB2GRAY)

    blob = cv2.dnn.blobFromImage(preprocessed_frame, 1. / 255, (28, 28))
    net.setInput(blob)
    preds = net.forward()
    label_idx = np.array(preds)[0].argmax()
    print('Predicted:', labels[label_idx])

    cv2.imshow('input', preprocessed_frame)
    cv2.imshow('test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
