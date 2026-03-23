import cv2
import numpy as np
import matplotlib.pyplot as plt

gray_image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)

edge_map = cv2.Canny(gray_image, 100, 150)

detected_lines = cv2.HoughLines(edge_map, 1, np.pi / 180, 100)

output_img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

if detected_lines is not None:
    for entry in detected_lines:
        distance, angle = entry[0]

        cos_val = np.cos(angle)
        sin_val = np.sin(angle)

        base_x = cos_val * distance
        base_y = sin_val * distance

        offset = 1000
        pt1 = (
            int(base_x + offset * (-sin_val)),
            int(base_y + offset * (cos_val))
        )
        pt2 = (
            int(base_x - offset * (-sin_val)),
            int(base_y - offset * (cos_val))
        )

        cv2.line(output_img, pt1, pt2, (0, 0, 255), 2)

rgb_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title("Detected Lines")
plt.axis("off")
plt.show()