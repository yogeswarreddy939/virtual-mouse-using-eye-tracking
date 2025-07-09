import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize webcam and Mediapipe FaceMesh
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Variables for multi-blink detection
blink_count = 0
last_blink_time = time.time()

# Variables for scroll control
scroll_threshold_up = 0.4
scroll_threshold_down = 0.6

# Main loop
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        # Move the cursor based on eye landmarks
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = int(landmark.x * screen_w)
                screen_y = int(landmark.y * screen_h)
                pyautogui.moveTo(screen_x, screen_y)
        
        # Detect blinks and implement multi-blink patterns
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        
        if (left[0].y - left[1].y) < 0.004:  # Blink detected
            if time.time() - last_blink_time > 0.5:  # Reset count if too long
                blink_count = 0
            blink_count += 1
            last_blink_time = time.time()

            if blink_count == 1:  # Single blink
                pyautogui.click()
                print("Single blink: Left Click")
            elif blink_count == 2:  # Double blink
                pyautogui.rightClick()
                print("Double blink: Right Click")
            elif blink_count == 3:  # Triple blink
                pyautogui.hotkey('win')
                print("Triple blink: Open Start Menu")
                blink_count = 0  # Reset count after triple blink

        # Reset blink count after a while
        if time.time() - last_blink_time > 1.0:
            blink_count = 0

        # Eye tracking for scroll control
        gaze_y = landmarks[145].y  # Using left eye's vertical landmark
        if gaze_y < scroll_threshold_up:  # Look up
            pyautogui.scroll(10)
            print("Scrolling up")
        elif gaze_y > scroll_threshold_down:  # Look down
            pyautogui.scroll(-10)
            print("Scrolling down")

    # Display the frame with feedback
    cv2.imshow('Eye Controlled Mouse', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
