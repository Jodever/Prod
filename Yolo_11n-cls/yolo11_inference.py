import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import image_processor
from image_processor import image_resize

# Load COCO pretrained YOLO11n model
model = YOLO("/home/jotroniks/Documents/Final-year-project/Prod/Yolo_11n-cls/train/weights/best.pt")


# Run inference with the YOLO11n model on test image
def prediction(processed_image):
    results = model.predict(processed_image, save=False)

    # for r in results:
        # Access predicted classes and confidence scores
        #print(f"Predicted class: {r.names[r.probs.top1]} (Confidence: {r.probs.top1conf:.2f})")
    result = results[0]
    inference = f"Predicted claass: {result.names[result.probs.top1]}\nConfidence: {result.probs.top1conf:.2f}"

    return inference



