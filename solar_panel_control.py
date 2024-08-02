import serial
import time
import csv
from datetime import datetime
import logging
import tensorflow as tf
import numpy as np

from inference import SolarPanelSAC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained SAC model
actor_model_path = 'trained_model/actor.h5'
actor_model = tf.keras.models.load_model(actor_model_path)

# Configure the serial connection
ser = serial.Serial(
    port='/dev/cu.usbmodem11201',  # Adjust this to your serial port
    baudrate=9600,
    timeout=1
)

def read_serial():
    try:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            return data
    except Exception as e:
        logger.error(f"[ERROR] Error reading from serial port: {str(e)}")
    return None

def send_command(command):
    try:
        ser.write((command + '\n').encode('utf-8'))
        response = read_serial()
        if response:
            logger.info(f"[INFO] Response: {response}")
        else:
            logger.warning("[WARNING] No response or timeout.")
    except Exception as e:
        logger.error(f"[ERROR] Error writing to serial port: {str(e)}")

def display_values():
    send_command("2000")
    while True:
        data = read_serial()
        if data:
            logger.info(f"[DATA] Received: {data}")
            return data
        time.sleep(0.1)

def move_servos(pos1, pos2):
    command = f"1000{pos1},{pos2}>"
    send_command(command)

def parse_data(data):
    try:
        values = data.split(',')
        if len(values) == 9:
            voltage, current, power, servo1, servo2, ldr1, ldr2, ldr3, ldr4 = map(float, values)
            return [ldr1, ldr2, ldr3, ldr4, voltage, current, power, servo1, servo2]
    except Exception as e:
        logger.error(f"[ERROR] Error parsing data: {str(e)}")
    return None

def save_to_csv(data):
    try:
        with open('solar_panel_data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().isoformat()] + data)
    except Exception as e:
        logger.error(f"[ERROR] Error saving to CSV: {str(e)}")

def inferencei(state):
    try:
        model_path = 'trained_model'
        agent = SolarPanelSAC(model_path)
        
        # Ensure state is in the correct format (9-dimensional numpy array)
        state = np.array(state, dtype=np.float32)
        if state.shape != (9,):
            raise ValueError("State should be a 9-dimensional array")
        
        # Get action from the model
        action = agent.get_action(tf.expand_dims(state, 0))
        
        # Ensure action is within expected range
        if np.any(action < -1) or np.any(action > 1):
            raise ValueError("Action out of expected range [-1, 1]")
        
        # Convert action from [-1, 1] to [0, 180] for servo angles
        new_angles = (action + 1) * 90
        
        # Adjust for the servo angle corruption
        new_angles[0] = min(new_angles[0] + 100, 180)
        
        logger.info(f"[DECISION] New angles calculated: {new_angles}")
        return new_angles.tolist()
    except Exception as e:
        logger.error(f"[ERROR] Error in inference: {str(e)}")
        return None

def test_model():
    try:
        model_path = 'trained_model'
        agent = SolarPanelSAC(model_path)
        # Test with a sample state
        sample_state = np.array([0.5, 0.5, 0.5, 0.5, 12.0, 1.0, 12.0, 90.0, 90.0], dtype=np.float32)
        action = agent.get_action(tf.expand_dims(sample_state, 0))
        logger.info(f"Sample action: {action}")
    except Exception as e:
        logger.error(f"[ERROR] Error testing model: {str(e)}")

if __name__ == "__main__":
    # Give some time for the connection to establish
    time.sleep(2)
    logger.info("Solar Panel Control System")
    
    test_model()  # Test model with a sample input

    try:
        current_angles = [90, 90]  # Initial angles
        while True:
            data = display_values()
            parsed_data = parse_data(data)
            logger.info(f"Parsed Data: {parsed_data}")
            
            if parsed_data:
                state = parsed_data
                save_to_csv(state)
                logger.info(f"State: {state}")
                
                new_angles = inferencei(state)
                logger.info(f"New Angles: {new_angles}")
                if new_angles:
                    move_servos(int(new_angles[0]), int(new_angles[1]))
                    current_angles = [new_angles[0], new_angles[1]]
                else:
                    logger.warning("[WARNING] Failed to get new angles from inference")
            else:
                logger.warning("[WARNING] Failed to parse data from Arduino")
            
            time.sleep(3)  # Adjust as needed

    except KeyboardInterrupt:
        logger.info("[INFO] Program interrupted by user.")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
    finally:
        ser.close()
        logger.info("[INFO] Serial connection closed.")
