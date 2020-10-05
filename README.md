Using OpenCV and HaarCascade Algorithm implemented a Face detection python program capable of detecting frontal human face other that that it can also detect if human is smiling.
In order to run program Webcam is needed, after importing all files in a folder just run command python Face-Detector.py.
This program was built on detecting face on webcam but we can easily detect faces on a pre recorded video just import video file in the folder and in Face Detector.py whereien webcam=cv2.videoCapture(0) is used just add your video link for eg: cv2.VideoCapture(vikings.mp4).
We can also detect face on image and recognise if it's smiling or not through similar process as above, i have commented down the code for Face Detection in images right after the Face Detection in webcam in Face-Detector.py feel free to check it out.
I have included a text document illustrating of how HaarCascade algorithm is implemented to get the desired results feel free to check that too.
