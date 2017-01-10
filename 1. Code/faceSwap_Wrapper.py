# Face Replacement Wrapper code:
# Instructions:
# Change the cv2.VideoCapture input with the video you want to do
# Face replacement on.
# Change cv2.VideoWriter input to the file name you want to save as.
# Change filename of the picture with which you want to replace the face in video
# This will work only with OpenCV version 3.0 and above

import sys
import numpy as np
import cv2
import dlib

import calculateDT
import warpAndBlend

global img1
global img2

if __name__ == '__main__' :
        
    # Video set-up
    cap = cv2.VideoCapture('easy1.avi')
    ret, frame = cap.read()
    height, width, channels = frame.shape
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('easy1_motioncomp.avi',fourcc, fps, (width, height))

    detector = dlib.get_frontal_face_detector() #Face detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

    # Read images
    filename1 = 'kk2.jpg'
    
    img1 = cv2.imread(filename1);

    
    # Get the face coordinates in Image1
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    points1 = []

    detections = detector(clahe_image, 1) #Detect the faces in the image
    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates
        for i in range(1,68): #There are 68 landmark points on each face
            points1.append((shape.part(i).x, shape.part(i).y))

    while True:
        ret, img2 = cap.read()
        if ret == True:
            print ret
            Warped_img1 = np.copy(img2);        
            # Get the face coordinates in Image2
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_image = clahe.apply(gray)
            
            
            detections = detector(clahe_image, 2) #Detect the faces in the image
            for k,d in enumerate(detections): #For each detected face
                shape = predictor(clahe_image, d) #Get coordinates
                points2 = []
                for i in range(1,68): #There are 68 landmark points on each face
                    points2.append((shape.part(i).x, shape.part(i).y))
                                
                # Convex hull
                hull_1 = []
                hull_2 = []

                hull_ind = cv2.convexHull(np.array(points2), returnPoints = False)
                      
                for i in xrange(0, len(hull_ind)):
                    hull_1.append(points1[hull_ind[i]])
                    hull_2.append(points2[hull_ind[i]])
                
                
                # Find delanauy traingulation for convex hull points
                img2_size = img2.shape    
                rect = (0, 0, img2_size[1], img2_size[0])
                 
                d_tri = calculateDT.calculateDelaunayTriangles(rect, points2)

                # Check if the number of triagles are zero
                if len(d_tri) == 0:
                    quit()
                
                # Apply affine transformation to Delaunay triangles
                for i in xrange(0, len(d_tri)):
                    tri_1 = []
                    tri_2 = []
                    
                    #get points for img1, img2 corresponding to the triangles
                    for j in xrange(0, 3):
                        tri_1.append(points1[d_tri[i][j]])
                        tri_2.append(points2[d_tri[i][j]])
                    
                    warpAndBlend.warpTriangle(img1, Warped_img1, tri_1, tri_2)
                
                        
                # Calculate Mask
                hull8U = []
                for i in xrange(0, len(hull_2)):
                    hull8U.append((hull_2[i][0], hull_2[i][1]))
                
                mask = np.ones(img2.shape, dtype = img2.dtype)  
                
                cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
                
                r = cv2.boundingRect(np.float32([hull_2]))    
                
                center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
                    
                
                # Clone seamlessly.
                output = cv2.seamlessClone(np.uint8(Warped_img1), img2, mask, center, cv2.MIXED_CLONE)
                img2 = output
            cv2.imshow("Face Swapped", output)
            out.write(output)
            if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
                break
        else:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
