import serial
import time
import csv
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os

# Configure the serial connection
ser = serial.Serial(
    port='/dev/cu.usbmodem11201',  # Update this to the correct port for your Pi
    baudrate=9600,
    timeout=1
)

# Set up logging
logging.basicConfig(filename='sac_evaluation.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SolarPanelSAC:
    def __init__(self, model_path):
        self.actor = tf.keras.models.load_model(os.path.join(model_path, 'actor'))

    def get_action(self, state):
        mean, log_std = self.actor(state)
        action = tf.tanh(mean)
        return action.numpy()[0]

def read_serial():
    try:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            return data
    except Exception as e:
        logging.error(f"[ERROR] Error reading from serial port: {str(e)}")
    return None

def send_command(command):
    try:
        ser.write((command + '\n').encode('utf-8'))
        response = read_serial()
        if response:
            logging.info(f"[INFO] Response: {response}")
        else:
            logging.warning("[WARNING] No response or timeout.")
    except Exception as e:
        logging.error(f"[ERROR] Error writing to serial port: {str(e)}")

def get_sensor_data():
    send_command("2000")
    data = read_serial()
    if data:
        try:
            values = list(map(float, data.split(',')))
            if len(values) == 7:
                return values
        except ValueError:
            logging.error(f"[ERROR] Invalid data format: {data}")
    return None

def move_servos(pos1, pos2):
    command = f"1000{pos1},{pos2}>"
    send_command(command)
    logging.info(f"[ACTION] Moved servos to positions: ({pos1}, {pos2})")

def save_to_csv(data):
    try:
        with open('sac_evaluation_data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().isoformat()] + data)
    except Exception as e:
        logging.error(f"[ERROR] Error saving to CSV: {str(e)}")

def generate_graphs(times, powers, angles):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(times, powers)
    plt.title('Power Production Over Time')
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.gcf().autofmt_xdate()
    
    plt.subplot(2, 1, 2)
    plt.plot(times, [a[0] for a in angles], label='Servo 1')
    plt.plot(times, [a[1] for a in angles], label='Servo 2')
    plt.title('Servo Angles Over Time')
    plt.xlabel('Time')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('sac_evaluation_results.png')
    plt.close()

def sac_evaluation(duration_minutes):
    model_path = 'trained_model'
    agent = SolarPanelSAC(model_path)
    
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    times = []
    powers = []
    angles = []
    total_power = 0
    current_angles = [90, 90]  # Starting position

    try:
        while datetime.now() < end_time:
            data = get_sensor_data()
            if data:
                ldr_values, voltage, current, power = data[:4], data[4], data[5], data[6]
                total_power += power
                
                state = ldr_values + [voltage, current, power] + current_angles
                state_array = np.array(state, dtype=np.float32)
                
                new_angles = agent.get_action(tf.expand_dims(state_array, 0))
                new_angles = (new_angles + 1) * 90  # Convert from [-1, 1] to [0, 180]
                new_angles[0] = min(new_angles[0] + 100, 180)  # Adjust for servo angle corruption
                
                move_servos(int(new_angles[0]), int(new_angles[1]))
                current_angles = new_angles.tolist()

                save_to_csv(data + current_angles)
                
                current_time = datetime.now()
                times.append(current_time)
                powers.append(power)
                angles.append(current_angles)
                
                logging.info(f"[DATA] Time: {current_time}, LDR: {ldr_values}, "
                             f"Power: {power}, New Angles: {current_angles}")
            
            time.sleep(60)  # Adjust panel every minute

    except KeyboardInterrupt:
        logging.info("[INFO] Evaluation interrupted by user.")
    except Exception as e:
        logging.error(f"[ERROR] Unexpected error during evaluation: {str(e)}")
    finally:
        ser.close()
        logging.info("[INFO] Serial connection closed.")
        
    generate_graphs(times, powers, angles)
    logging.info(f"[RESULT] Total power produced: {total_power} W")
    print(f"Total power produced: {total_power} W")
    print(f"Evaluation data saved to 'sac_evaluation_data.csv'")
    print(f"Graphs saved to 'sac_evaluation_results.png'")

if __name__ == "__main__":
    try:
        duration = int(input("Enter the duration for SAC evaluation (in minutes): "))
        sac_evaluation(duration)
    except ValueError:
        logging.error("[ERROR] Invalid input for duration. Please enter an integer.")
        print("Invalid input. Please enter a valid number of minutes.")
    except Exception as e:
        logging.error(f"[ERROR] Unexpected error: {str(e)}")
        print(f"An unexpected error occurred. Please check the log file for details.")