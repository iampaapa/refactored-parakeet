import serial
import time
import csv
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

# Configure the serial connection
ser = serial.Serial(
    port='/dev/ttyACM0',  # Update this to the correct port for your Pi
    baudrate=9600,
    timeout=1
)

# Set up logging
logging.basicConfig(filename='fixed_angle_control_evaluation.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

def save_to_csv(data):
    try:
        with open('fixed_angle_evaluation.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().isoformat()] + data)
    except Exception as e:
        logging.error(f"[ERROR] Error saving to CSV: {str(e)}")

def generate_graph(times, powers):
    plt.figure(figsize=(10, 6))
    plt.plot(times, powers)
    plt.title('Power Production Over Time (Fixed Angle)')
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.gcf().autofmt_xdate()
    plt.savefig('fixed_angle_power.png')
    plt.close()

def fixed_angle_test(angle1, angle2, duration_minutes):
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    times = []
    powers = []
    total_power = 0

    try:
        # Set the fixed angle
        move_servos(angle1, angle2)
        logging.info(f"[INFO] Set panel to fixed angle: ({angle1}, {angle2})")

        while datetime.now() < end_time:
            data = get_sensor_data()
            if data:
                power = data[6]
                total_power += power
                
                save_to_csv(data + [angle1, angle2])
                
                current_time = datetime.now()
                times.append(current_time)
                powers.append(power)
                
                logging.info(f"[DATA] Time: {current_time}, Power: {power}")
            
            time.sleep(60)  # Record data every minute

    except KeyboardInterrupt:
        logging.info("[INFO] Program interrupted by user.")
    except Exception as e:
        logging.error(f"[ERROR] Unexpected error: {str(e)}")
    finally:
        ser.close()
        logging.info("[INFO] Serial connection closed.")
        
    generate_graph(times, powers)
    logging.info(f"[RESULT] Total power produced: {total_power} W")
    print(f"Total power produced: {total_power} W")

if __name__ == "__main__":
    angle1 = int(input("Enter the angle for servo 1 (0-180): "))
    angle2 = int(input("Enter the angle for servo 2 (0-180): "))
    duration = int(input("Enter the duration for fixed angle test (in minutes): "))
    fixed_angle_test(angle1, angle2, duration)