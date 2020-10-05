### Steps for Face Detetection ###
# Step 1 - Getting xml file of thousands of faces
# Step 2 - change all those images in xml file to grayscale (no rgb)
# Step 3 - Train out ML alogorithm with converted grayscale images

### Steps for detecting smile in faces ###
# Step 1 - Find faces in our image(use above face detection algorithm(Haar Algorithm))
# Step 2 - Find smile within those faces(Haar Algo)
# Step 3 - Label the faces if it's smiling

### Real Time Face detection through webcam ###
import cv2

# 1st step
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
trained_smiling_face_data = cv2.CascadeClassifier('haarcascade_smile.xml')

# Instead of image we will be using webcam/recorded video, here in VideoCapture(0) we passed paramter as 0 since we have only one camera, in case if there are multiple webcams connected then we can use 1,2 etc for eg Videocapture(1) indicates 2nd camera and VideCapture(2) represents 3rd camera and so on
# we can use also pass video to VideoCapture() by simply impoerting video file to our current directory and passing vide0 file like this VideoCapture(video-file)

# to capture vide from webcamer using webcam variable
webcam = cv2.VideoCapture(0)

# Iteratng over every frame of webcam(also can be done for the videos), it is an infinite loop since we want to keep detecting faces unless webcam is shut down

while True:

    # Reading the current frame
    # read function will return two parameters:
    # successful_frame_read - 1st being, whether reading the frame was successful/not so it will essentially return a boolean(true/false)
    # frame -  2nd being, the actual frame(screenshot/image)
    successful_frame_read, frame = webcam.read()

    # we are breaking the loop if frame read is unsuccessful due to some interruption
    if not successful_frame_read:
        break

    # 2nd step converting current frame to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3rd step- drawing rectangle around all faces in current frame
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # to make a sub image(that sub image being only faces in the frame), we are using slicing here and slicing being 2D here from x co-ordinate we want to slice from x to x+w and from y co-ordinate we want to slice from y  to y+h
        faces_in_frame = frame[y:y+h, x:x+w]
        face_grayscale = cv2.cvtColor(faces_in_frame, cv2.COLOR_BGR2GRAY)

        # here scaleFactor means by how much factor we want to blur them frame so facial features can be properly identified(in order to avoid randomly detecting other things), minNeighbours means within the face if there are 20 rectangle focusing on few pixels then it is probably a face
        smiles = trained_smiling_face_data.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # Finding all smiles in the face by drawing a rectangle
        # for(x_smile, y_smile, w_smile, h_smile) in smiles:
        #     cv2.rectangle(faces_in_frame, (x_smile, y_smile),
        #                   (x_smile+w_smile, y_smile+h_smile), (255, 0, 0), 2)

        # Finding all smiles in the face by putting a text smiling
        # the way the logic works is if in current frame a scmile is detected the value stored in smiles variable will be greated than 0 otherwise it wil be 0 where fontScale - size of the font, fontFace - type of font(times new roman etc), color - color of the font
        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 0))

    cv2.imshow('Face Detector', frame)
    # if we don't press any button, key will store 0 otherwise a non-negatve value will be stored in key if we press a key
    key = cv2.waitKey(1)
    # incase if we want to exit real time face detection, by closing webcam, we can simply break infinite while loop by writing following piece of code, it simply means if user presses q/Q it will break while loop here we are using ASCII values of q(81) and Q(113)

    if key == 81 or key == 113:
        break

# we are releasing VideoCapture() from webcam as it is good practice(similar to flushing pointers in C)
webcam.release()
# we are closing all windows as good practice
cv2.destroyAllWindows()

### Testing face detection in images ###
"""
import cv2

# To load some pre trained data on face frontals from opencv using haarcascadealgorithm, here xml contain all preteined frontal faces, using this pretrained frontal faces we can detect a new face by comparing features
trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

# imread is used for reading images, we are storing read value in variable img
img = cv2.imread('Ragnar.jpg')

# we convert all those images to grayscale, here cvttColor will convert our image to desire color we want in this case we are converting our image i.e. RGB/BGR to Grayscale. to cvtColor we pass 1st parameter the input image(for which we will be changing color) and second parameter will be color to which we want to change the image(in this case gray(COLOR_BGR2GRAY))
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# using our pretrained frontal faces xml we  will identify new faces, in this case our converted greyscale image,here detected features that covers face are consolidated and returned as coordinates of a rectangle(top left) and width,depth to be added to the top left coordinate to get a full rectangle, here if images has multiple faces then array of multiple values in form (x,y,width,height) will be returned
# so basically detectMultiScale will return array of objects with each object in the form (x,y,width,height) where x,y is co-ordinate to the top left
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# returns the array of objects containg face co-ordinates objects, i.e. part covering the face
# print(face_coordinates)

# after getting rectangle coordinates and widths,depths to draw rectangle around face we use opencv command rectangle wherein first parameter will now be our colored image(on which we want to detect faces),2nd parameters will be coordinates of rectangle(top left), 4th parameter will be width and depth we need to add to the top left co-ordinate to get proper rectangle around our face, 5th parameter will be color of reactangles(BGR) and 6th parameter will be thickness of the edges
# we will be looping through each face(object(x,y,width,height)) in the image and drawing rectangle around them
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Using show command we are displaying face images with the title-'Face detector'
cv2.imshow('Face Detector', img)
cv2.waitKey()  # cv2.imshow will open a new window and show image but it will be displayed for a split second and then it will automatically close, so in order for it to be displayed for a longer time we use cv2.waitkey(), it will display the image for as long as we want and once we want to close the image and execute rest of code, we just press any button in keyboard and it will automatically close it and the rest of code execution resumes normally
#If we want waitKey() tio terminates after certain amount of time without pressing a key, we can simply pass a number as a parameter indicating for how many seconds waitkey() should display image untile a key is pressed
# for eg if waitkey(10)- her image will be displayed for 10 milliseconds and then it will abort displaying the image and continue with normal execution of code, we can also abort image early by pressing a key before 10 milliseconds and it will simply abort the image display and continue with normal execution of code


print("code completed")
"""
