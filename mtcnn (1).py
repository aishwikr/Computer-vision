import cv2
from mtcnn.mtcnn import MTCNN
#from PIL import Image 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

detector = MTCNN()

image = cv2.imread("ActiOn_1.jpg")
result = detector.detect_faces(image)

for i in range(0,len(result)):
    bounding_box = result[i]['box']
    keypoints = result[i]['keypoints']
    
    #print(bounding_box)
    #print(keypoints)
    
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
    
    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    
cv2.imwrite("img_drawn.jpg", image)
img=mpimg.imread('img_drawn.jpg')
imgplot = plt.imshow(img)
plt.show()

print(result)


"""Webcam Code as follows: """

cap = cv2.VideoCapture(0)
while True: 
    #Capture frame-by-frame
    __, frame = cap.read()
    
    #Use MTCNN to detect faces
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            for i in range(0,len(result)):
                bounding_box = person[i]['box']
                keypoints = person[i]['keypoints']
        
                cv2.rectangle(frame,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                              (0,155,255),
                              2)
        
                cv2.circle(frame,(keypoints['left_eye']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['right_eye']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['nose']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_left']), 2, (0,155,255), 2)
                cv2.circle(frame,(keypoints['mouth_right']), 2, (0,155,255), 2)
    #display resulting frame
    cv2.imshow('image',frame)
    
    if cv2.waitKey(1) &0xFF == ord('q'):
        break

#When everything's done, release capture
cap.release()
cv2.destroyAllWindows()