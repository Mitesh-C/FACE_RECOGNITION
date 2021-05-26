import cv2
import os
import numpy as np
import facerecognition as fr

test_img = cv2.imread(r"C:\Users\Mitesh Choksi\Desktop\FaceRecognition\Testimg\others4.jpg")
faces_detected,gray_img = fr.faceDetection(test_img)
print("Faces Detected = ",faces_detected)

#for (x,y,w,h) in faces_detected:
#cv2.rectangle(test_img, (x,y), (x+w,y+h),(0,255,0),thickness=5)

#resized = cv2.resize(test_img,(1000,700))
#cv2.imshow("Face Detection", resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#faces, faceID = fr.labels_for_training_data(r'C:\Users\Mitesh Choksi\Desktop\FaceRecognition\TraningImg')
#face_recognizer = fr.train_classifier(faces, faceID)
#face_recognizer.save('trainingData.yml')
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'C:\Users\Mitesh Choksi\Desktop\FaceRecognition\\trainingData.yml')
names = {0:"OTHER HEROS", 1:"MITESH"}
for face in faces_detected:
    (x,y,w,h) = face
    roi_gray=gray_img[y:y+w , x:x+h]
    label, confidense = face_recognizer.predict(roi_gray)
    print("CONFIDENSE :: ", confidense)
    print("LABEL :: ", label)
    fr.draw_rectangle(test_img, face)
    pred_name = names[label]
    fr.put_text(test_img,pred_name, x, y)

resized = cv2.resize(test_img,(1000,700))
cv2.imshow("Face Detection", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()