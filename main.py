import cv2

img = cv2.imread("/Users/jed.ryan/VisualStudioProjects/PythonProject/assets/skateboard.webp", cv2.IMREAD_GRAYSCALE)
print(img.shape)
# cv2.imshow("Skateboard", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
