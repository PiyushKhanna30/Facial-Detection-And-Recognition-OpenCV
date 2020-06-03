import cv2
face_cascade=cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
img=cv2.imread('my_images/image.jpg')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(img_gray,scaleFactor=1.5,minNeighbors=4)
for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imwrite("my_images/OutputImages/facesDetected.jpg",img)
cv2.imshow("Frame",img)
if cv2.waitKey(0)==ord('q'):
	cv2.destroyAllWindows()
