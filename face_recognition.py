from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os


# import numpy as np

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(self.root, text="FACE RECOGNITION", font=("times new roman", 35, "bold"), bg="Chocolate",
                          fg="Black")
        title_lbl.place(x=0, y=0, width=1450, height=45)

        img = Image.open(r"C:\Users\Pratik\Desktop\FACE-REGONITION\College_Image\8.jpg")
        img = img.resize((2200, 896), Image.ANTIALIAS)
        self.photoimg = ImageTk.PhotoImage(img)

        f_lb1 = Label(self.root, image=self.photoimg)
        f_lb1.place(x=0, y=40, width=2000, height=900)

        b0_1 = Button(f_lb1, text="Face Recognition", cursor="hand2", font=("times new roman", 15, "bold"),
                      command=self.face_recog, bg="chocolate", fg="Black", borderwidth=7)
        b0_1.place(x=620, y=600, width=160, height=40)

    # ==========face recognition============
    def face_recog(self):
        print("hello")

        def draw_boundary(img2, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

            coord = []

            for (x, y, w, h) in features:
                cv2.rectangle(img2, [int(x), int(y)], (x + w, y + h), (0, 255, 0), 3)
                id, predict = clf.predict(gray_image[y:y + h, x:x + w])
                confidence = int((100 * (1 - predict / 300)))

                conn = mysql.connector.connect(host="localhost", user="root", password="", database="face_recognizer")
                my_cursor = conn.cursor()

                my_cursor.execute("select Name from student where Student_id=" + str(id))
                n = my_cursor.fetchone()
                n = "+".join(n)

                my_cursor.execute("select Roll from student where Student_id=" + str(id))
                r = my_cursor.fetchone()
                r = "+".join(r)

                my_cursor.execute("select Dep from student where Student_id=" + str(id))
                d = my_cursor.fetchone()
                d = "+".join(d)

                if confidence > 77:
                    cv2.putText(img2, f"Roll:{r}", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img2, f"Name:{n}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                    cv2.putText(img2, f"Department:{d}", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                else:
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 3)
                    cv2.putText(img2, "Unknown Face", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

                coord = [x, y, w, y]
            return coord

        def recognize(img1, clf, faceCascade):
            coord = draw_boundary(img1, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
            return img1

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_cap = cv2.VideoCapture(0)

        while True:
            ret, faceimg = video_cap.read()
            faceimg = recognize(faceimg, clf, faceCascade)
            cv2.imshow("Welcome To Face Recognition", faceimg)

               # to exit face_recognizer press "Q" key and close face_recognizer window
            key = cv2.waitKey(20)
            if key == ord('q'):
                break
                video_cap.release()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition(root)
    root.mainloop()
