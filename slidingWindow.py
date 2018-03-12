from imutils import resize
import cv2

image = cv2.imread("Testimages/test4.png")
image = resize(image, height=min(180, image.shape[1]))

(h, w) = image.shape[:2]
print(h, w)

for i in range(0, h, 64):
    for j in range(0, w, 64):
        print(i, j)
        cv2.rectangle(image, (j, i), (j+64, i+128), (255, 0, 0))
        cv2.imshow("Sliding window test", image)
        cv2.waitKey(0)