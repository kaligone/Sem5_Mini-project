from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
from student import Student
import os
from train import Train
from face_recognition import Face_Recognition

class Face_Recognition_System:
    def __init__(self,root):
        self.root=root
        self.root.attributes('-fullscreen', False)
        self.root.geometry("1550x800+0+0")
        self.root.title("Face Recognition System")

        #College Image
        img=Image.open(r"C:\Users\Pratik\Desktop\FACE-REGONITION\College_Image\1.jpeg")
        img=img.resize((2000,900),Image.ANTIALIAS)
        self.photoimg=ImageTk.PhotoImage(img)

        f_lbl=Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=40,width=2000,height=900)


        #title
        title_lbl=Label(text="FACE RECOGNITION BASED ATTENDANCE SYSTEM",font=("times new roman",35,"bold"),fg="Black")
        title_lbl.place(x=0,y=0,width=1450,height=45)

        #student button
        img1=Image.open(r"C:\Users\Pratik\Desktop\FACE-REGONITION\College_Image\2.jpg")
        img1=img1.resize((500,220),Image.ANTIALIAS)
        self.photoimg1=ImageTk.PhotoImage(img1)

        b1=Button(f_lbl,image=self.photoimg1,command=self.student_details,cursor="hand2")
        b1.place(x=150,y=50,width=500,height=220)

        b1_1=Button(f_lbl,text="Student Details",command=self.student_details,cursor="hand2",font=("times new roman",15,"bold"),bg="chocolate",fg="Black")
        b1_1.place(x=150,y=270,width=500,height=40)




        #Detect face button
        img2=Image.open(r"C:\Users\Pratik\Desktop\FACE-REGONITION\College_Image\3.jpg")
        img2=img2.resize((500,220),Image.ANTIALIAS)
        self.photoimg2=ImageTk.PhotoImage(img2)

        b2=Button(f_lbl,image=self.photoimg2,cursor="hand2",command=self.face_data)
        b2.place(x=800,y=50,width=500,height=220)

        b2_1=Button(f_lbl,text="Face Detector",cursor="hand2",command=self.face_data,font=("times new roman",15,"bold"),bg="chocolate",fg="Black")
        b2_1.place(x=800,y=270,width=500,height=40)



        #Attendance  button
        img3=Image.open(r"C:\Users\Pratik\Desktop\FACE-REGONITION\College_Image\4.jpg")
        img3=img3.resize((500,220),Image.ANTIALIAS)
        self.photoimg3=ImageTk.PhotoImage(img3)

        b3=Button(f_lbl,image=self.photoimg3,cursor="hand2")
        b3.place(x=150,y=400,width=500,height=220)

        b3_1=Button(f_lbl,text="Attendance",cursor="hand2",font=("times new roman",15,"bold"),bg="chocolate",fg="Black")
        b3_1.place(x=150,y=620,width=500,height=40)


        #Train Data  button
        img4=Image.open(r"C:\Users\Pratik\Desktop\FACE-REGONITION\College_Image\5.jpg")
        img4=img4.resize((500,220),Image.ANTIALIAS)
        self.photoimg4=ImageTk.PhotoImage(img4)

        b4=Button(f_lbl,image=self.photoimg4,cursor="hand2",command=self.train_data)
        b4.place(x=800,y=400,width=500,height=220)

        b4_1=Button(f_lbl,text="Train Data",cursor="hand2",command=self.train_data,font=("times new roman",15,"bold"),bg="chocolate",fg="Black")
        b4_1.place(x=800,y=620,width=500,height=40)


        #Photo
        b5_1=Button(f_lbl,text="Photo",cursor="hand2",command=self.open_img,font=("times new roman",15,"bold"),bg="chocolate",fg="Black",borderwidth = 7)
        b5_1.place(x=680,y=330,width=80,height=40)


    def open_img(self):
        os.startfile("data")
 #function btn

    def student_details(self):
        self.new_window=Toplevel(self.root)
        self.app=Student(self.new_window)

    def train_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Train(self.new_window)

    def face_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Face_Recognition(self.new_window)


       


if __name__ == "__main__":
    root=Tk()
    obj=Face_Recognition_System(root)
    root.mainloop()


