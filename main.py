import cv2
import numpy as np
import Adafruit_PCA9685
import time

# Initialize the servo motor
pwm = Adafruit_PCA9685.PCA9685(0x41)
pwm.set_pwm_freq(50)

# Function to set the servo angle
def set_servo_angle(channel, angle):
    angle = 4096 * ((angle * 11) + 500) / 20000
    pwm.set_pwm(channel, 0, int(angle))

# Initial servo positions
X_P = 90
Y_P = 90
set_servo_angle(1, X_P)
set_servo_angle(2, Y_P)

# Camera settings
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set frame width
cap.set(4, 480)  # Set frame height

# Define the light green color range in HSV
green_lower = np.array([35, 100, 100])  # Adjust based on the exact shade of green
green_upper = np.array([85, 255, 255])

# Control sensitivity for servo response
control_sensitivity = 8

# PID parameters
Kp = 0.155  # Proportional gain
Ki = 0.011  # Integral gain
Kd = 0.155  # Derivative gain

# Variables for PID control
integral_x = 0
integral_y = 0
previous_error_x = 0
previous_error_y = 0

# Variables to track timing and states
start_time = None
has_centered = False

# Helper function to implement PID control
def pid_control(error, integral, previous_error, Kp, Ki, Kd):
    integral += error
    derivative = error - previous_error
    output = Kp * error + Ki * integral + Kd * derivative
    return output, integral

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV and mask for green color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours for potential circular green clusters
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    # Initialize variable to store the center of detected pattern
    pattern_center = None
   
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:  # Only consider significant contours
            hull = cv2.convexHull(c)
            if len(hull) > 5:  # More points indicate a circular shape
                (x, y), radius = cv2.minEnclosingCircle(hull)
                if 20 < radius < 50:  # Adjust radius range for your pattern size
                    pattern_center = (int(x), int(y))
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)
                    cv2.circle(frame, pattern_center, int(radius), (255, 0, 0), 2)
                    break  # Stop if pattern found

    # Move the servos based on the detected pattern center
    if pattern_center:
        cx, cy = pattern_center
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        # Calculate the difference between the pattern center and the image center
        dx = cx - frame_center_x
        dy = cy - frame_center_y

        # Start timing when movement begins, but only if not already centered
        if not has_centered and start_time is None:
            start_time = time.time()
            print("Green object detected, starting timer.")

        # Apply PID control for smoother, non-oscillating servo movement
        pid_output_x, integral_x = pid_control(dx, integral_x, previous_error_x, Kp, Ki, Kd)
        pid_output_y, integral_y = pid_control(dy, integral_y, previous_error_y, Kp, Ki, Kd)

        # Adjust the servo angles using PID output
        X_P -= pid_output_x / control_sensitivity
        Y_P -= pid_output_y / control_sensitivity

        # Limit the angles within bounds
        X_P = max(0, min(175, X_P))
        Y_P = max(0, min(175, Y_P))

        # Set servo angles for pan and tilt
        set_servo_angle(1, X_P)
        set_servo_angle(2, Y_P)

        # Check if the object is centered
        if abs(dx) < 10 and abs(dy) < 10 and not has_centered:  # Allowable margin for centering
            elapsed_time = time.time() - start_time
            if elapsed_time >= 0.01:  # Ensure timing is above a threshold
                print("Time taken to center: {:.2f} seconds".format(elapsed_time))
            else:
                print("Ignored timing as it's too fast.")
            has_centered = True  # Mark as centered
            start_time = None  # Reset start time for next detection
            previous_error_x = 0
            previous_error_y = 0
    else:
        # Reset state when the green object disappears
        if has_centered:
            print("Object lost. Ready for next detection.")
        has_centered = False
        start_time = None
        previous_error_x = 0
        previous_error_y = 0

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
