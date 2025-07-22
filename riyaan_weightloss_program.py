import cv2
import numpy as np
import os

def main():
    """
    Main function to run real-time color and object detection.
    """
    # --- Object Detection Setup (Haar Cascade for Face Detection) ---
    # Construct the path to the Haar Cascade file. This is a more robust
    # way to locate it within the OpenCV package.
    casc_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    
    # Check if the cascade file exists
    if not os.path.exists(casc_path):
        print(f"Error: Cascade file not found at {casc_path}")
        return
        
    face_cascade = cv2.CascadeClassifier(casc_path)
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier.")
        return
    
    # --- Camera Setup ---
    # Open the default camera (camera index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Camera opened successfully. Press 'q' to quit.")

    # --- Main Loop ---
    while True:
        # Read a new frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # 1. First, perform color detection on the frame
        frame_with_color_detection = detect_color(frame)

        # 2. Then, perform face detection on the result
        final_frame = detect_faces(frame_with_color_detection, face_cascade)

        # Display the resulting frame with all detections
        cv2.imshow('Object and Color Detection', final_frame)

        # Check for user input to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")

def detect_faces(frame, face_cascade):
    """
    Detects faces in a frame using a Haar Cascade classifier.

    Args:
        frame: The input frame.
        face_cascade: The loaded CascadeClassifier for face detection.

    Returns:
        The frame with rectangles drawn around detected faces.
    """
    # Haar cascades work better on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Red rectangle

    return frame

def detect_color(frame):
    """
    Processes a single frame to detect objects of a specific color.

    Args:
        frame: The input frame from the camera (in BGR color space).

    Returns:
        A frame with detected color objects highlighted.
    """
    # Convert the frame from BGR to HSV color space
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for the color to be detected (Blue)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Create a binary mask
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours of the white areas in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around the detected contours
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a green rectangle for color detection
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

if __name__ == '__main__':
    main()
