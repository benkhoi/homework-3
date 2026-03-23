import cv2
import matplotlib.pyplot as plt

gray = cv2.imread('test.jpg', 0)

edge_result = cv2.Canny(gray, threshold1=100, threshold2=200)

plt.imshow(edge_result, cmap='gray')
plt.title('Edge Detection Result')
plt.xticks([])
plt.yticks([])
plt.show()