# Standard imports
import cv2
import numpy as np 
 
# Read images
src = cv2.imread("aeroplane.jpg")
dst = cv2.imread("scenery.jpg")
 
cv2.imshow('Plane',src)
cv2.imshow('Mountains',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


src_nr, src_nc, src_ncol = src.shape
print src_nr, src_nc, src_ncol

dst_nr, dst_nc, dst_ncol = dst.shape
print dst_nr, dst_nc, dst_ncol


# Create a rough mask around the airplane.
src_mask = np.zeros(src.shape, src.dtype)

poly = np.array([ [0,0], [160,85], [0,85], [160,0] ], np.int32)
src_mask[5:82,1:160] = (255,255,255)
#cv2.fillPoly(src_mask, [poly], (255, 255, 255))

cv2.imshow('src_mask',src_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# This is where the CENTER of the airplane will be placed
center = (200,80)
 
# Clone seamlessly.
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

cv2.imshow('Output',output)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
# Save result
cv2.imwrite("scenery-plane.jpg", output);
