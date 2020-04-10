import cv2
import numpy as np
import matplotlib.pyplot as plt


# Affine Transform

# Image load as Single precision(8 bit index, 23 bit log)
img = cv2.imread('D:/Others/Programming/Project Space/Franklin_Clinton/Images/Frank0017.jpg').astype(np.float32) / 255.0
# Declaration of Variable(Rows, Columns, Channel)
rows, cols, ch = img.shape
# Image size(Rows * Columns * Channel)
print(img.size)

# Set the Coordinate of Three Points as Single precision
pts1 = np.float32([[56.25, 56.25], [168.75, 56.25], [56.25, 168.75]])
pts2 = np.float32([[28.125, 28.125], [196.875, 28.125], [28.125, 196.875]])
# Slicing in Pts List
pts_x, pts_y = zip(*pts1)
# Display 3 Points in Image
plt.imshow(img, cmap=plt.cm.bone)
plt.scatter(pts_x, pts_y, c='w', s=100)
# Show Image
plt.show()

# Calculation Homography Matrix for Affine Transform
H = cv2.getAffineTransform(pts1, pts2)
H

# Peform the Affine Transform
img2 = cv2.warpAffine(img, H, (cols, rows))

# Zoom Image
x_pts, y_pts = zip(*pts2)
plt.imshow(img2, cmap=plt.cm.bone)
plt.scatter(x_pts, y_pts, c='w', s=100)
plt.xlim(0, 225)
plt.ylim(225, 0)
plt.show()


# Edge Detection

# Peform the Edge Detection
edges = cv2.Canny(np.uint8(img2 * 255), 50, 100)

# Show Image
plt.imshow(edges)
plt.show()
