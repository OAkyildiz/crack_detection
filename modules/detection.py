#!/usr/bin/env python

#import numpy as np
import cv2
import threading
from os import getcwd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def predictWithModel(model,frame):

  reshape(frame) #sliding window or scaling

  #lbl, loc=model.predict(frame)
  return lbl, loc

def markDetected(mat,loc):
  pass

def onlinePrediction(model,source=0, show_source=True, output='output.avi'):
  cap = cv2.VideoCapture(source)
  h,w=int(cap.get(3)),int(cap.get(4))
  #source != 0
  if source:
    fps=cap.get(7)
    T=int(1000/fps) if fps else 30
  else:
    fps=30
    T=int(1000/30)

  fourcc = cv2.VideoWriter_fourcc(*'XVID')# I want .h264. Maybe parameterize
  out = cv2.VideoWriter(output,fourcc, fps, (h,w)) #Thread 3, #rename out,output
  #thread inits?
  cv2.namedWindow('Predictor Output')
  cv2.namedWindow('Source Video')
  cv2.moveWindow('Predictor Output', 800, 0)

  #

  while(True):
    # Capture frame-by-frame
    ret, rawframe = cap.read()
    if ret==True:
        # Thread 1:
        #lbl, loc=predictWithModel(rawframe)
        lbl=False
        if lbl:
          processed=markDetected(rawframe,loc)
        else: #potentiall combine these  in mark detected.
          processed=rawframe ### THIS IS NOT HIS FINAL FORM. ineffective
          cv2.putText(processed, 'X', (w-20,h-10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0))

        cv2.imshow('Predictor Output',processed)
        # Thread 2:
        cv2.imshow('Source Video',rawframe)
         # Thread 3? (or 1):
        out.write(processed)

        if cv2.waitKey(int(1)) & 0xFF == ord('q'):
          break

        # When everything done, release the capture
  cap.release()
  out.release()
  cv2.destroyAllWindows()

##    cv2.moveWindow

def onlineMultiModelPrediciton():
  #thinking of using multiple predictors to test at once.
  pass

def main():
  vid='/home/oakyildiz/workspaces/directed_research/crack_detection/target_data/cracks_test_vid.mp4'
  onlinePrediction(0,vid)

if __name__ == "__main__":
      exit(main())
