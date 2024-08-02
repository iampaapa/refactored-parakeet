import serial
import time
import csv
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt

# Configure the serial connection
ser = serial.Serial(
    port='/dev/cu.usbmodem11201',  # Update this to the correct port for your Pi
    baudrate=9600,
    timeout=1
)

# Set up logging
logging.basicConfig(filename='light_chasing.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

current_pos = [90, 90]  # Starting position for servos

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
            return response
        else:
            logging.warning("[WARNING] No response or timeout.")
    except Exception as e:
        logging.error(f"[ERROR] Error writing to serial port: {str(e)}")

def get_sensor_data():
    data = send_command("2000")
    # data = read_serial()
    if data:
        try:
            values = list(map(float, data.split(',')))
            if len(values) == 9:

                return values
        except ValueError:
            logging.error(f"[ERROR] Invalid data format: {data}")
    return None

def move_servos(pos1, pos2):
    command = f"1000{pos1},{pos2}>"
    send_command(command)

def normalize_readings(readings):
    minResistance = [34, 37, 0, 60]
    maxResistance = [990, 992, 1006, 1009]
    normalizedValues = [(readings[i] - minResistance[i]) / (maxResistance[i] - minResistance[i]) * 100 for i in range(4)]
    return normalizedValues

def chase_light(ldr_values):
    global current_pos
    postop, posbase = current_pos
    step = 1

    if ldr_values[0] > ldr_values[3]:
        postop += step
    if ldr_values[2] > ldr_values[1]:
        postop += step
    if ldr_values[3] > ldr_values[0]:
        postop -= step
    if ldr_values[1] > ldr_values[2]:
        postop -= step
    if ldr_values[0] > ldr_values[2]:
        posbase -= step
    if ldr_values[3] > ldr_values[1]:
        posbase -= step
    if ldr_values[1] > ldr_values[3]:
        posbase += step
    if ldr_values[2] > ldr_values[0]:
        posbase += step

    postop = max(0, min(180, postop))
    posbase = max(0, min(180, posbase))

    return postop, posbase

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
    
    print("Starting light chasing mode...")

    try:
        # Activate light chasing mode
        send_command("3333")
        
        while datetime.now() < end_time:
            data = get_sensor_data()
            if data:
                # Extract power and LDR readings
                power = data[2]
                ldr_readings = data[-4:]
                
                # Append to totals and lists
                total_power += power
                times.append(datetime.now())
                powers.append(power)
                
                # Save to CSV with current position
                save_to_csv(data + list(current_pos))
                
                logging.info(f"[DATA] Time: {times[-1]}, LDR: {ldr_readings}, Power: {power}, Position: {current_pos}")

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
