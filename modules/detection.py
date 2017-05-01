#!/usr/bin/env python

#import numpy as np
import cv2
import threading

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

def predictWithModel(model,frame):
    
  reshape(frame) #sliding window or scaling
    
  lbl=model.predict(frame)
  return lbl, loc
    
def markDetected(mat,loc):
	pass

def onlinePrediction(self,model,source=0, show_source=True, output='output.avi'):
    cap = cv2.VideoCapture(0)
    h,w=cap.get(3),cap.get(4)
	if source: #source != 0
		fps=cap.get(7)
		T=int(1000/fps)
	else:
		fps=30
		T=int(1000/30)
		
	fourcc = cv2.VideoWriter_fourcc(*'XVID')# I want .h264. Maybe parameterize
	out = cv2.VideoWriter(output,fourcc, fps, (h,w)) #Thread 3
	#thread inits?
	cv2.namedWindow('Predictor Output')
	cv2.namedWindow('Source Video')
	cv2.moveWindow('Predictor Output', 800, 0) 
	
	#
	
    while(cap.isOpened()):
        # Capture frame-by-frame
      ret, rawframe = cap.read()
      if ret==True:
        # Thread 1:
        lbl, loc=predictWithModel(rawframe)
        
        if lbl:
        	processed=markDetected(rawframe,loc)
        else: #potentiall combine these  in mark detected.
			processed=rawframe ### THIS IS NOT HIS FINAL FORM. ineffective
        	cv2.putText(processed, 'X', (w-20,h-10), cv2.HERSHEY_PLAIN, 2, (255,0,0))
        	
        cv2.imshow('Predictor Output',marked)
        # Thread 2:
        cv2.imshow('Source Video',rawframe)
       	# Thread 3? (or 1):
       	output.write(processed)
       	
        if cv2.waitKey(T/2) & 0xFF == ord('q'):
          break
		
        # When everything done, release the capture
    cap.release()
    output.release()
    cv2.destroyAllWindows()

##    cv2.moveWindow

def onlineMultiModelPrediciton():
	#thinking of using multiple predictors to test at once.
	pass
