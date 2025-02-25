# Computer Vision-->OpenCV

![Python Logo](https://www.python.org/static/community_logos/python-logo.png)

## 🔥 Introduction
Welcome to this repository! Here, you'll find a collection of practical OpenCV codes for image and video processing, along with essential operations in computer vision and data analysis.

---

## 🛠️ Requirements
Before running the code, make sure you have the required libraries installed:
bash
pip install opencv-python matplotlib numpy


---

## 📂 Course Contents
### 🖼️ Image Processing with OpenCV

#### 📌 Reading & Displaying Images
python
import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("path/to/image.jpg")

# Display image using OpenCV
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Display image using Matplotlib
plt.imshow(img)
plt.axis('off')
plt.show()


#### 🎨 Color Space Conversion
python
# Convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Image', gray_img)
cv2.waitKey(0)

# Convert image to HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', hsv_img)
cv2.waitKey(0)


#### ✂️ Resizing & Cropping Images
python
# Resize the image
resized_img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)

# Crop a specific region of the image
cropped_img = img[20:220, 150:293]  # (height, width)
cv2.imshow('Cropped Image', cropped_img)
cv2.waitKey(0)


#### 🔍 Applying Image Effects
python
# Apply Gaussian Blur
blur_img = cv2.GaussianBlur(img, (9,9), 0)
cv2.imshow('Blurred Image', blur_img)
cv2.waitKey(0)

# Edge detection using Canny
edges = cv2.Canny(blur_img, 75, 150)
cv2.imshow('Edges', edges)
cv2.waitKey(0)


---

### 🎥 Video Processing with OpenCV
python
# Read video
capture = cv2.VideoCapture("path/to/video.mp4")

while True:
    isTrue, frame = capture.read()
    if not isTrue:
        break
    cv2.imshow('Video', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


---

### 🔲 Drawing Shapes & Adding Text
python
import numpy as np

# Create a blank image
blank = np.zeros((500,500,3), dtype='uint8')

# Draw a rectangle
cv2.rectangle(blank, (50,50), (200,200), (0,255,0), thickness=2)

# Draw a circle
cv2.circle(blank, (250,250), 50, (255,0,0), thickness=3)

# Add text
cv2.putText(blank, 'AI & ML', (50,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

cv2.imshow('Shapes & Text', blank)
cv2.waitKey(0)


---

## 🚀 What's Next?
This repository will be continuously updated with more hands-on examples and real-world applications in **Machine Learning & OpenCV**. Stay tuned for more exciting content! 🎯

🔗 **Follow & Star** this repository to keep up with the latest updates!
