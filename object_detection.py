
import cv2
import numpy as np
import os

def get_target_color():
    """Prompts the user to enter the color they want to detect."""
    while True:
        # Prompt user for input
        color_input = input("Enter the color to detect ('red' or 'blue'): ").lower()
        if color_input in ['red', 'blue']:
            return color_input
        else:
            print("Invalid input. Please enter 'red' or 'blue'.")

def detect_all_colors(frame):
    """
    Detects all configured colors in a single frame.

    Args:
        frame: The input frame from the camera.

    Returns:
        A dictionary where keys are color names and values are lists of contours found.
    """
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected_objects = {'red': [], 'blue': []}

    # --- Color Ranges Definition ---
    color_ranges = {
        'red': [
            {'lower': np.array([0, 120, 70]), 'upper': np.array([10, 255, 255])},
            {'lower': np.array([170, 120, 70]), 'upper': np.array([180, 255, 255])}
        ],
        'blue': [
            {'lower': np.array([100, 150, 50]), 'upper': np.array([140, 255, 255])}
        ]
    }

    for color_name, ranges in color_ranges.items():
        # Combine masks if a color has multiple ranges (like red)
        final_mask = None
        for r in ranges:
            mask = cv2.inRange(hsv_image, r['lower'], r['upper'])
            if final_mask is None:
                final_mask = mask
            else:
                final_mask = cv2.bitwise_or(final_mask, mask)

        # Morphological operations to remove noise
        kernel = np.ones((7, 7), np.uint8)
        clean_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)

        # Find and store contours
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                detected_objects[color_name].append(contour)
    
    return detected_objects


def draw_detection_info(frame, contour, color_name):
    """
    Draws the bounding box and info for a single detected object.

    Args:
        frame: The frame to draw on.
        contour: The contour of the detected object.
        color_name: The name of the color detected.
    """
    box_color = (0, 0, 255) if color_name == 'red' else (255, 200, 0)
    frame_height = frame.shape[0]

    rect = cv2.minAreaRect(contour)
    (x, y), (width, height), angle = rect

    # --- Elevation Calculation ---
    # Y=0 is top, Y=frame_height is bottom. We flip it for intuitive percentage.
    elevation_percent = int((1 - (y / frame_height)) * 100)

    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(frame, [box], 0, box_color, 2)

    # --- Prepare text for display ---
    if width < height:
        angle += 90
        width, height = height, width
    if angle > 45:
        angle -= 90

    dim_text = f"W: {int(width)} H: {int(height)}"
    angle_text = f"Angle: {int(angle)} deg"
    elevation_text = f"Elevation: {elevation_percent}%"

    # --- Display the text ---
    text_pos_y = int(y) - 10
    cv2.putText(frame, dim_text, (int(box[1][0]), int(box[1][1] - 35)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, angle_text, (int(box[1][0]), int(box[1][1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, elevation_text, (int(box[1][0]), int(box[1][1] + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    """
    Main function to run the real-time detection application.
    """
    target_color = get_target_color()
    print(f"Looking for '{target_color}' objects. Press 'q' to quit.")

    # --- Camera Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # --- Main Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect all colors in the frame
        all_detections = detect_all_colors(frame)

        # Check if any non-target colors were found
        other_color_found = False
        for color, contours in all_detections.items():
            if color != target_color and len(contours) > 0:
                other_color_found = True
                break

        # Display warning if other colors are present
        if other_color_found:
            cv2.putText(frame, "Don't pick it up!", (50, 50),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)

        # Draw info for the target color objects
        if all_detections[target_color]:
            for contour in all_detections[target_color]:
                draw_detection_info(frame, contour, target_color)

        # Display the resulting frame
        cv2.imshow('Advanced Color and Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed.")


if __name__ == '__main__':
    main()

