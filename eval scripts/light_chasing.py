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
logging.basicConfig(filename='light_chasing.log', level=logging.DEBUG,
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

def chase_light(ldr_values):
    # Simple light chasing algorithm
    # Adjust these thresholds based on your specific setup
    threshold = 10
    step = 5

    horizontal_diff = (ldr_values[0] + ldr_values[1]) - (ldr_values[2] + ldr_values[3])
    vertical_diff = (ldr_values[0] + ldr_values[2]) - (ldr_values[1] + ldr_values[3])

    new_pos1 = max(0, min(180, current_pos[0] + (step if horizontal_diff > threshold else -step if horizontal_diff < -threshold else 0)))
    new_pos2 = max(0, min(180, current_pos[1] + (step if vertical_diff > threshold else -step if vertical_diff < -threshold else 0)))

    return new_pos1, new_pos2

def save_to_csv(data):
    try:
        with open('light_chasing_data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().isoformat()] + data)
    except Exception as e:
        logging.error(f"[ERROR] Error saving to CSV: {str(e)}")

def generate_graph(times, powers):
    plt.figure(figsize=(10, 6))
    plt.plot(times, powers)
    plt.title('Power Production Over Time')
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.gcf().autofmt_xdate()
    plt.savefig('light_chasing_power.png')
    plt.close()

def light_chasing_control(duration_minutes):
    global current_pos
    current_pos = [90, 90]  # Starting position
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    times = []
    powers = []
    total_power = 0

    try:
        while datetime.now() < end_time:
            data = get_sensor_data()
            if data:
                ldr_values = data[:4]
                power = data[6]
                total_power += power
                
                new_pos = chase_light(ldr_values)
                move_servos(*new_pos)
                current_pos = new_pos

                save_to_csv(data + list(current_pos))
                
                current_time = datetime.now()
                times.append(current_time)
                powers.append(power)
                
                logging.info(f"[DATA] Time: {current_time}, LDR: {ldr_values}, Power: {power}, Position: {current_pos}")
            
            time.sleep(1)  # Adjust this delay as needed

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
    duration = int(input("Enter the duration for light chasing (in minutes): "))
    light_chasing_control(duration)