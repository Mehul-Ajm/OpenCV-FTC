import cv2
import numpy as np


# --- 1. CALIBRATION & SETUP ---


# !!! IMPORTANT !!!
# MEASURE THE WIDTH OF ONE OF YOUR BLOCKS IN CENTIMETERS AND UPDATE THIS VALUE
KNOWN_BLOCK_WIDTH_CM = 5.0


# This is a sample focal length. It's an estimate and may need to be adjusted
# for your specific camera for better accuracy. A higher value means the camera
# has a "zoomed-in" effect.
# You can recalibrate this by:
# 1. Running the script.
# 2. Holding a block at a known distance (e.g., 30 cm) from the camera.
# 3. Note the reported "Pixel Width".
# 4. Calculate: New_Focal_Length = (Pixel_Width * Known_Distance_cm) / KNOWN_BLOCK_WIDTH_CM
SAMPLE_FOCAL_LENGTH = 840


# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# Get the center of the camera frame (once)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_center_x = frame_width // 2
frame_center_y = frame_height // 2


cv2.namedWindow('Block Position Tracking')
print("Starting video stream. Press 'q' or close the window to quit.")




# --- 2. HELPER FUNCTIONS ---


def distance_to_camera(known_width, focal_length, pixel_width):
    """Calculates the distance from the camera to a known object."""
    if pixel_width == 0:
        return 0
    return (known_width * focal_length) / pixel_width


def find_and_draw_blocks(frame, hsv_frame, lower_bound, upper_bound, color_name):
    """
    Finds blocks of a specific color, calculates their XYZ position, and draws on the frame.
    """
    mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)


    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500: # Filter out small noise
            x_box, y_box, w_box, h_box = cv2.boundingRect(largest_contour)


            # A. Calculate Distance (Z-coordinate)
            distance_cm = distance_to_camera(KNOWN_BLOCK_WIDTH_CM, SAMPLE_FOCAL_LENGTH, w_box)


            # B. Calculate Real-world X, Y coordinates
            # Find the center of the block in pixel coordinates
            block_center_x = x_box + w_box // 2
            block_center_y = y_box + h_box // 2


            # Calculate pixel offset from the frame center
            pixel_offset_x = block_center_x - frame_center_x
            pixel_offset_y = block_center_y - frame_center_y


            # Convert pixel offset to real-world cm offset
            # (Similar triangles principle)
            x_cm = (pixel_offset_x * distance_cm) / SAMPLE_FOCAL_LENGTH
            y_cm = (pixel_offset_y * distance_cm) / SAMPLE_FOCAL_LENGTH


            # C. Draw information on the screen
            box_color = (0, 255, 0) # Green for all boxes
            cv2.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), box_color, 2)


            # Display coordinates
            coord_text_x = f"X: {x_cm:.2f} cm"
            coord_text_y = f"Y: {y_cm:.2f} cm"
            coord_text_z = f"Z: {distance_cm:.2f} cm (Dist)"


            cv2.putText(frame, color_name, (x_box, y_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            cv2.putText(frame, coord_text_x, (x_box, y_box + h_box + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            cv2.putText(frame, coord_text_y, (x_box, y_box + h_box + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            cv2.putText(frame, coord_text_z, (x_box, y_box + h_box + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            cv2.putText(frame, f"Pixel Width: {w_box}", (x_box, y_box + h_box + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)




# --- 3. MAIN APPLICATION LOOP ---


while True:
    ret, frame = cap.read()
    if not ret:
        break


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # HSV color ranges
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
   
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
   
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])


    # Find each block
    # For red, we need to combine two masks
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    full_red_mask = red_mask1 + red_mask2
    # Since find_and_draw_blocks needs the HSV frame, we'll handle red separately
    # This is a small simplification for clarity
    find_and_draw_blocks(frame, hsv, lower_blue, upper_blue, "Blue")
    find_and_draw_blocks(frame, hsv, lower_yellow, upper_yellow, "Yellow")
   
    # Process the combined red mask
    contours_red, _ = cv2.findContours(full_red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_red) > 0:
        largest_red_contour = max(contours_red, key=cv2.contourArea)
        # Re-use the drawing/calculation logic, but with the red contour
        if cv2.contourArea(largest_red_contour) > 500:
            x_r, y_r, w_r, h_r = cv2.boundingRect(largest_red_contour)
            distance_cm_r = distance_to_camera(KNOWN_BLOCK_WIDTH_CM, SAMPLE_FOCAL_LENGTH, w_r)
            x_cm_r = ((x_r + w_r // 2) - frame_center_x) * distance_cm_r / SAMPLE_FOCAL_LENGTH
            y_cm_r = ((y_r + h_r // 2) - frame_center_y) * distance_cm_r / SAMPLE_FOCAL_LENGTH
           
            box_color_r = (0, 0, 255) # Red for the box
            cv2.rectangle(frame, (x_r, y_r), (x_r + w_r, y_r + h_r), box_color_r, 2)
            cv2.putText(frame, "Red", (x_r, y_r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color_r, 2)
            cv2.putText(frame, f"X: {x_cm_r:.2f} cm", (x_r, y_r + h_r + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_r, 2)
            cv2.putText(frame, f"Y: {y_cm_r:.2f} cm", (x_r, y_r + h_r + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_r, 2)
            cv2.putText(frame, f"Z: {distance_cm_r:.2f} cm (Dist)", (x_r, y_r + h_r + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_r, 2)




    # Display the final frame
    cv2.imshow('Block Position Tracking', frame)


    # Quit logic
    key_press = cv2.waitKey(1) & 0xFF
    if key_press == ord('q') or cv2.getWindowProperty('Block Position Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break


# --- 4. Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Video stream stopped.")




