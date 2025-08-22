import cv2
import numpy as np

size = 224

def image_resize(uploaded_image):
     file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
     
     img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
     
     processed_image = cv2.resize(img, (224,224))
     
     return processed_image

