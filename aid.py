import cv2  
from ultralytics import YOLO 

# Load YOLO model
model = YOLO("yolov5s.pt")  # Use "yolov5s.pt" or "yolov8n.pt"
model.to("cuda") 
vid = 'fellas.mp4'  # Path to the video file
cam = cv2.VideoCapture(vid)

while cam.isOpened():
    det, frame = cam.read()
    
    if det:
        results = model.predict(frame, device=0)  # Perform detection
        
        for r in results:
            for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                x1, y1, x2, y2 = map(int, box)  # Bounding box coordinates
                class_id = int(cls)  # Get class ID
                label = model.names[class_id]  # Get class name
                confidence = float(conf)  # Get confidence score

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display label above the object
                label_text = f"{label} {confidence:.2f}"
                cv2.putText(frame, 
                            label_text, 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0), 
                            2, 
                            cv2.LINE_AA)

        cv2.imshow("YOLO Detection", frame)  # Show frame with detections

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()