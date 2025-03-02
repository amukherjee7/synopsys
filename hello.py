import serial
import time

# Change 'COM3' to your Arduino port (e.g., '/dev/tty.usbmodemXXXX' on Mac)
port = '/dev/tty.usbmodem1101'
baud_rate = 9600  # Same as in Arduino sketch
file_name = 'data.csv'

# Open serial port
ser = serial.Serial(port, baud_rate)
time.sleep(2)  # Wait for Arduino to reset

# Open file to write data
with open(file_name, 'w') as f:
    f.write('A1,A2\n')  # Write CSV header

    # Read data for 60 seconds
    start_time = time.time()
    while time.time() - start_time < 60:
        line = ser.readline().decode('utf-8').strip()  # Read a line from Arduino
        f.write(line + '\n')  # Write the line to the file

# Close the serial port
ser.close()
print(f'Data saved to {file_name}')