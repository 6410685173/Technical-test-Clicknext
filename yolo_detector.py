from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2 as cv
from collections import defaultdict
import torch 
# Load YOLO model
model = YOLO("yolov8n.pt")
device = 'cuda' if torch.cuda.is_available() else 'cpu' # It will be faster if we can install torch that supports cuda
model.to(device)
track_box_history = defaultdict(lambda: []) # store centroid point in each object (but it will have only cat class now)

def draw_boxes(frame, boxes):
    """Draw detected bounding boxes on image frame"""

    # Create annotator object
    annotator = Annotator(frame)

    for box in boxes:
        class_id = box.cls
        class_name = model.names[int(class_id)]
        coordinator = box.xyxy[0]
        x_centroid, y_centroid, h, w = box.xywh[0]
        confidence = box.conf

        track = track_box_history[15] # use classid represent trackid (I tried using box id but sometimes box id is None)
        track.append((float(x_centroid), float(y_centroid)))
        
        # Draw bounding box
        annotator.box_label(
            box=coordinator, label=class_name, color=(255,0,0)
        )

        # Draw centroid point and track trails
        if len(track)>15: # plot the track point only 15 points backward.
            del track[0]  
        annotator.draw_centroid_and_tracks(track, color=(255, 0, 0), track_thickness=1)
    return annotator.result()


def detect_object(frame):
    """Detect object from image frame"""

    # Detect object from image frame
    results = model.track(frame, classes=[15], persist=True, tracker="bytetrack.yaml") #detect only cat (cls 15 is the cat class in YOLO from ultralytics)
    for result in results:
        frame = draw_boxes(frame, result.boxes)

    return frame


if __name__ == "__main__":
    video_path = "CatZoomies.mp4"
    cap = cv.VideoCapture(video_path)
    
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    text = "Nipitpon-Kampolrat-Clicknext-Internship-2024"
    while cap.isOpened():
        # Read image frame
        ret, frame = cap.read()

        if ret:
            # Detect cat from image frame
            frame_result = detect_object(frame)
            cv.putText(frame_result, text, (frame_width-len(text)*14,20), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
            

            # Show result
            cv.namedWindow("Video", cv.WINDOW_NORMAL)
            cv.imshow("Video", frame_result)
            cv.waitKey(30)

        else:
            break

    # Release the VideoCapture object and close the window
    
    cap.release()
    cv.destroyAllWindows()
