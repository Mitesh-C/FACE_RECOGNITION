import cv2
import os
import numpy as np


def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier(
        r"C:\Users\Mitesh Choksi\Desktop\FaceRecognition\HaarCascade\haarcascade_frontalface_default.xml")
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=2)
    return faces, gray_img


def labels_for_training_data(directory):
    faces = []
    faceID = []
    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("SKIPPING SYSTEM FILES...")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("\nIMAGE PATH:", img_path)
            print("IMAGE ID: ", id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("IMAGE NOT LOADED PROPERLY!!!")
                continue
            faces_detect, gray_img = faceDetection(test_img)
            if len(faces_detect) != 1:
               # print("Faces detected are not exactly 1...")
                continue  # Since we are expecting the imageds with one faces
            (x, y, w, h) = faces_detect[0]
            roi_gray = gray_img[y:y + w, x:x + h]  # Y ROI BCOZ to get Region Of Interest to get gray scal image
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID

def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create() # LBPH = Local Binary Pattern Histogram
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer

def draw_rectangle(test_img,face):
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w,y+h),(0,255,0),thickness=1)

def put_text(test_img,text,x,y):
    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),1)




#MAIN CODE
test_img = cv2.imread(r"C:\Users\Mitesh Choksi\Desktop\FaceRecognition\Testimg\mitesh4.jpg")
faces_detected,gray_img = faceDetection(test_img)
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
    draw_rectangle(test_img, face)
    pred_name = names[label]
    put_text(test_img,pred_name, x, y)

resized = cv2.resize(test_img,(1000,700))
cv2.imshow("Face Detection", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()





