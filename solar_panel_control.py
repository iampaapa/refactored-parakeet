import serial
import time
import csv
from datetime import datetime
import logging
from inference import inference, logger

# Configure the serial connection
ser = serial.Serial(
    port='/dev/ttyACM0',  # Update this to the correct port for your Pi
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
        if len(values) == 7:
            ldr1, ldr2, ldr3, ldr4, voltage, current, power = map(float, values)
            return [ldr1, ldr2, ldr3, ldr4, voltage, current, power]
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

if __name__ == "__main__":
    # Give some time for the connection to establish
    time.sleep(2)

    try:
        current_angles = [90, 90]  # Initial angles
        while True:
            data = display_values()
            parsed_data = parse_data(data)
            
            if parsed_data:
                state = parsed_data + current_angles
                save_to_csv(state)
                
                new_angles = inference(state)
                if new_angles:
                    logger.info(f"[DECISION] Moving servos to angles: {new_angles}")
                    move_servos(int(new_angles[0]), int(new_angles[1]))
                    current_angles = new_angles
                else:
                    logger.warning("[WARNING] Failed to get new angles from inference")
            else:
                logger.warning("[WARNING] Failed to parse data from Arduino")
            
            time.sleep(60)  # Wait for 1 minute before next adjustment

    except KeyboardInterrupt:
        logger.info("[INFO] Program interrupted by user.")
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {str(e)}")
    finally:
        ser.close()
        logger.info("[INFO] Serial connection closed.")