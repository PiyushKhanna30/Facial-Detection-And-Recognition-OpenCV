import cv2
import numpy as np
import pickle
import os

def rescale(frame,percent=30):
	dim=(550, 550)
	return cv2.resize(frame,dim)

face_cascade=cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()

name_ids_dict={}
name_id=0
img_name=0

x_train=[]
y_label=[]

base_dir=os.path.dirname(os.path.abspath(__file__))#get path where this file placed
image_dir=os.path.join(base_dir,"imagesToTrain")#base_dir/imagesToTrain

for root,dirs,files in os.walk(image_dir):
	#root : base_dir/imagesToTrain ,base_dir/imagesToTrain/folders
	#dirs : dirs in base_dir/imagesToTrain, base_dir/imagesToTrain/folder_name/
	#files : files in base_dir/imagesToTrain ,and so on
	for file in files:
		if(file.endswith('jpg') or file.endswith('png')):
			path=os.path.join(root,file)
			name=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
			if not name in name_ids_dict:#here inserting name into dict with name id
				name_ids_dict[name]=name_id
				name_id+=1
			n_id=name_ids_dict[name]# extracting id relative to name of person
			img=cv2.imread(path,0)# opening image in grayscale
			img=rescale(img,50)
			img_array=np.array(img,'uint8')
			faces=face_cascade.detectMultiScale(img_array,1.05,6)
			for (x,y,w,h) in faces:
				roi = img_array[x:x+w,y:y+h]
				n=str(img_name)+".png"
				cv2.imwrite(os.path.join("trainedData/imagesTrained",n),roi)
				img_name+=1
				x_train.append(roi)
				y_label.append(n_id)
				print(".",end=" ")
with open("trainedData/name.pickle","wb") as f:
	pickle.dump(name_ids_dict,f)
recognizer.train(x_train,np.array(y_label))
recognizer.save("trainedData/train.yml")
print(y_label)
print(name_ids_dict)