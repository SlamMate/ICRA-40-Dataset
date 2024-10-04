import pygame
import requests
import threading
import os
import time

# Define the URL for car control
control_url = 'http://localhost:8000/control'

# Define the log file path
log_file_path = 'control_log.csv'

# Create the log file if it doesn't exist
if not os.path.exists(log_file_path):
    with open(log_file_path, 'w') as log_file:
        log_file.write("# timestamp linear angular\n")

# Function to get current data from the bot
def get_current_data():
    try:
        response = requests.get('http://localhost:8000/data')
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Function to log the control command in TUM format (only timestamp, linear, and angular)
def log_command_tum_format(linear, angular):
    current_data = get_current_data()
    if current_data and "timestamp" in current_data:
        timestamp = current_data["timestamp"]  # Get the timestamp

        # TUM format logging, including only timestamp, linear, and angular
        log_entry = f"{timestamp} {linear} {angular}\n"

        # Write the log to the file
        with open(log_file_path, 'a') as log_file:
            log_file.write(log_entry)
        
        print(f"Logged data in TUM format: {log_entry.strip()}")
    else:
        print("Failed to log command: timestamp not found in car data.")

# Function to send control commands
def send_command(linear, angular):
    data = {
        "command": {
            "linear": linear,
            "angular": angular,
            "lamp": 0
        }
    }
    try:
        response = requests.post(control_url, json=data)
        if response.status_code == 200:
            print(f'Successfully sent command: linear={linear}, angular={angular}, lamp=0')
            log_command_tum_format(linear, angular)
        else:
            print(f'Failed to send command: {response.status_code}')
    except requests.exceptions.RequestException as e:
        print(f"Error sending command: {e}")

# Global variables to store the state
linear_state = 0
angular_state = 0
emergency_stop = False  # Indicates emergency stop status
command_thread = None  # Manages the command sending thread

# Thread control flags
thread_flags = {
    'up': threading.Event(),
    'down': threading.Event(),
    'left': threading.Event(),
    'right': threading.Event()
}

# Thread function to continuously send commands at 0.1-second intervals
def command_sender():
    global linear_state, angular_state, emergency_stop
    while not emergency_stop:
        send_command(linear_state, angular_state)
        time.sleep(0.1)  # Send command every 0.1 seconds

# Start the command sending thread
def start_command_sender():
    global command_thread
    if command_thread is None or not command_thread.is_alive():
        command_thread = threading.Thread(target=command_sender, daemon=True)
        command_thread.start()

# Initialize pygame and detect the joystick
def init_joystick():
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print("No joystick detected. Please connect a Joy-Con.")
        exit()
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Detected joystick: {joystick.get_name()}")
    return joystick

# Listen to joystick input
def control_car():
    global linear_state, angular_state, emergency_stop

    joystick = init_joystick()

    print("Use Joy-Con to control the car.")
    print("Move the left stick for forward/backward and rotation.")
    print("Press 'Minus' button for emergency stop. Press 'Plus' to resume. Press 'Home' to exit.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.JOYAXISMOTION:
                # Assume the left stick vertical controls linear, horizontal controls angular
                linear = -joystick.get_axis(1)  # Forward is negative
                angular = joystick.get_axis(0)   # Right is positive

                # Apply a deadzone
                deadzone = 0.1
                if abs(linear) < deadzone:
                    linear = 0
                if abs(angular) < deadzone:
                    angular = 0

                if not emergency_stop:
                    linear_state = linear
                    angular_state = angular
                    start_command_sender()

            elif event.type == pygame.JOYBUTTONDOWN:
                # Assume button mapping as follows:
                # Button 13: Minus (Emergency Stop)
                # Button 12: Plus (Resume)
                # Button 16: Home (Exit)
                button = event.button
                if button == 13:  # Minus
                    emergency_stop = True
                    linear_state = 0
                    angular_state = 0
                    send_command(0, 0)
                    print("Emergency stop activated!")
                elif button == 12:  # Plus
                    if emergency_stop:
                        emergency_stop = False
                        print("Exited emergency stop mode, resuming normal operation.")
                        start_command_sender()
                elif button == 16:  # Home
                    print("Exiting control...")
                    running = False

        time.sleep(0.01)

    pygame.quit()

if __name__ == '__main__':
    control_car()
