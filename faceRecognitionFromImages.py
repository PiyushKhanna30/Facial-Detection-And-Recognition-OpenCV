import cv2
import pickle

def rescale(frame):
	dim=(520,520)
	return cv2.resize(frame,dim)

face_cascade=cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainedData/train.yml')
with open ("trainedData/name.pickle","rb") as f:
	name_ids_dict=pickle.load(f)
	rev_name_ids_dict={v:k for k,v in name_ids_dict.items()}

img=cv2.imread('my_images/image.jpg')
# img=rescale(img)
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(img_gray,scaleFactor=1.05,minNeighbors=6)
for (x,y,w,h) in faces:
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
	name="Unknown"
	try:
		roi=img_gray[x:x+w,y:y+h]
		name_id,confidence=recognizer.predict(roi)
		print(name_id,confidence)
		if confidence<=100: #0 is perfect match
			name=rev_name_ids_dict[name_id]
			print(name)
	except:
		pass
	cv2.putText(img,name,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
cv2.imwrite("my_images/OutputImages/facesRecognised.jpg",img)
cv2.imshow("Frame",img)
if cv2.waitKey(0)==ord('q'):
	cv2.destroyAllWindows()