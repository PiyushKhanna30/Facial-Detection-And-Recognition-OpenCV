import cv2
face_cascade=cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")

cap=cv2.VideoCapture(0)
while cap.isOpened():
	ret,frame=cap.read()
	if ret:
		gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces=face_cascade.detectMultiScale(gray_frame,scaleFactor=1.5,minNeighbors=1)
		for (x,y,w,h) in faces:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.imshow("Frame",frame)
		if cv2.waitKey(1)==ord('q'):
			break
	else:
		break
cap.release()
cv2.destroyAllWindows()

# #How haar files work?
# #Haar cascade classifier works with a sliding window approach. 
# #If you look at the cascade files you can see a size parameter which 
# # usually a pretty small value like 20 20. This is the smallest window 
# # that cascade can detect. So by applying a sliding window approach, 
# # you slide a window through out the picture than you resize it and search 
# # again until you can not resize it further. So with every iteration 
# # haar's cascaded classifier true outputs are stored. So when this 
# # window is slided in picture resized and slided again; it actually 
# # detects many many false positives. You can check what it detects by 
# # giving minNeighbors 0. 

# # So there are a lot of face detection because of resizing the sliding 
# # window and a lot of false positives too. So to eliminate false 
# # positives and get the proper face rectangle out of detections, 
# # neighborhood approach is applied. It is like if it is in neighborhood 
# # of other rectangles than it is ok, you can pass it further. So this 
# # number determines the how much neighborhood is required to pass it 
# # as a face rectangle. See by setting minNeighbors 1

# # So by increasing this number you can eliminate false positives but be 
# # careful, by increasing it you can also lose true positives too.
# # See by setting minNeighbors 3~6 i got good results

# # scaleFactor – Parameter specifying how much the image size is reduced at 
# # each image scale.

# # minSize – Minimum possible object size. Objects smaller than that are ignored.
# # maxSize – Maximum possible object size. Objects larger than that are ignored.

# #Sourses
# https://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale
# https://stackoverflow.com/questions/22249579/opencv-detectmultiscale-minneighbors-parameter?noredirect=1&lq=1