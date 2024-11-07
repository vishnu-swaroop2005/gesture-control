import cv2
import mediapipe as mp
import pyautogui
import screen_brightness_control as pct

# Initialize all coordinates
x1 = x2 = y1 = y2 = x3 = y3 = x4 = y4 = x5 = y5 = 0

webcam = cv2.VideoCapture(0)
my_hands = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    output = my_hands.process(rgb_image)
    hands = output.multi_hand_landmarks
    
    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark
            
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                # Track relevant landmarks
                if id == 8:  # Index fingertip
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x1, y1 = x, y
                if id == 4:  # Thumb tip
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 0, 255), thickness=3)
                    x2, y2 = x, y
                if id == 12:  # Middle fingertip
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x3, y3 = x, y
                if id == 16:  # Ring fingertip
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x4, y4 = x, y
                if id == 20:  # Pinky fingertip
                    cv2.circle(img=image, center=(x, y), radius=8, color=(0, 255, 255), thickness=3)
                    x5, y5 = x, y
        
        # Calculate distances
        dis1 = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 // 4  # Thumb to Index distance
        dis2 = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5 // 4  # Thumb to Pinky distance
        dis3 = ((x2 - x4) ** 2 + (y2 - y4) ** 2) ** 0.5 // 4
        dis4 = ((x2 - x5) ** 2 + (y2 - y5) ** 2) ** 0.5 // 4
        
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Line between thumb and index
        cv2.line(image, (x5, y5), (x2, y2), (0, 255, 0), 5)  # Line between thumb and pinky
        cv2.line(image, (x3, y3), (x2, y2), (0, 255, 0), 5)
        cv2.line(image, (x4, y4), (x2, y2), (0, 255, 0), 5)

        # Debug prints for distances
        print(f"dis1 (Thumb-Index): {dis1}, dis2 (Thumb-Pinky): {dis2}")

        # Volume and brightness control logic
        if dis1 > 20 and dis4<10:
            pyautogui.press("volumeup")
        elif dis1 < 20 and dis4 <10:
            pyautogui.press("volumedown")
            pct.set_brightness(20)
        else:
            pct.set_brightness(100)

        
    # Display the image
    cv2.imshow("Hand volume control using python", image)
    
    # Exit on 'Esc' key press
    key = cv2.waitKey(10)
    if key == 27:
        break

# Release the webcam and destroy windows
webcam.release()
cv2.destroyAllWindows()
