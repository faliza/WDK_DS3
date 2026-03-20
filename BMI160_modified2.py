import smbus
import math
import time
import numpy as np
import os
# Constants
BMI160_I2C_ADDR = 0x69
ACCEL_SENSITIVITY = 16384.0  # ±2g sensitivity for the accelerometer in LSB/g

bus = smbus.SMBus(1)
# Initialize I2C with specified pins
# i2c = I2C(0, scl=Pin(1), sda=Pin(0), freq=400000)

def write_register(addr, reg, data):
    """Write data to a register."""
    #i2c.writeto_mem(addr, reg, bytes([data]))
    bus.write_byte_data(addr, reg, data)

def read_register(addr, reg, length):
    """Read data from a register."""
    return bus.read_i2c_block_data(addr, reg, length)

def initialize_bmi160():
    """Initialize the BMI160 sensor."""
    # Set accelerometer to normal mode
    write_register(BMI160_I2C_ADDR, 0x7E, 0x11)  # ACC_NORMAL_MODE
    write_register(BMI160_I2C_ADDR, 0x7E, 0x15)  # GYR_NORMAL_MODE
    time.sleep(0.1)

def read_raw_acceleration():
    """Read raw acceleration data."""
    data = read_register(BMI160_I2C_ADDR, 0x12, 6)  # Read accel data
    ax_raw = int.from_bytes(data[0:2], 'little') - (1 << 16 if data[1] & 0x80 else 0)
    ay_raw = int.from_bytes(data[2:4], 'little') - (1 << 16 if data[3] & 0x80 else 0)
    az_raw = int.from_bytes(data[4:6], 'little') - (1 << 16 if data[5] & 0x80 else 0)
    return ax_raw, ay_raw, az_raw

def read_raw_gyroscope():
    """Read raw acceleration data."""
    data = read_register(BMI160_I2C_ADDR, 0x0C, 6)  # Read accel data
    ax_raw = int.from_bytes(data[0:2], 'little') - (1 << 16 if data[1] & 0x80 else 0)
    ay_raw = int.from_bytes(data[2:4], 'little') - (1 << 16 if data[3] & 0x80 else 0)
    az_raw = int.from_bytes(data[4:6], 'little') - (1 << 16 if data[5] & 0x80 else 0)
    return ax_raw, ay_raw, az_raw


def auto_calibrate():
    """Perform auto-calibration to remove noise or error."""
    print("Starting auto-calibration...")
    num_samples = 100
    ax_offset = 0
    ay_offset = 0
    az_offset = 0

    for _ in range(num_samples):
        ax_raw, ay_raw, az_raw = read_raw_acceleration()
        ax_offset += ax_raw
        ay_offset += ay_raw
        az_offset += az_raw
        time.sleep(0.01)  # Small delay between readings

    # Calculate average offsets
    ax_offset //= num_samples
    ay_offset //= num_samples
    az_offset //= num_samples

    # Assuming the sensor is stable, Z-axis should measure 1g (gravity)
    az_offset -= int(ACCEL_SENSITIVITY)

    print("Auto-calibration completed.")
    print("Offsets - X: {}, Y: {}, Z: {}".format(ax_offset, ay_offset, az_offset))

    return ax_offset, ay_offset, az_offset

def read_acceleration(ax_offset, ay_offset, az_offset):
    """Read raw acceleration data, apply offsets, and convert to m/s²."""
    ax_raw, ay_raw, az_raw = read_raw_acceleration()
    ax = ((ax_raw) / ACCEL_SENSITIVITY) * 9.81  # Convert to m/s²
    ay = ((ay_raw) / ACCEL_SENSITIVITY) * 9.81  # Convert to m/s²
    az = ((az_raw) / ACCEL_SENSITIVITY) * 9.81  # Convert to m/s²
    return ax, ay, az

def read_gyroscope(ax_offset, ay_offset, az_offset):
    """Read raw acceleration data, apply offsets, and convert to m/s²."""
    gx, gy, gz = read_raw_gyroscope()
    return gx, gy, gz

def calculate_tilt_angles(ax, ay, az):
    """Calculate pitch and roll angles from acceleration."""
    pitch = math.atan2(ay, math.sqrt(ax**2 + az**2)) * 180.0 / math.pi
    roll = math.atan2(-ax, az) * 180.0 / math.pi
    return pitch, roll

# Initialize BMI160
initialize_bmi160()
print("BMI160 Initialized")

def sensor_sleep(time_left):

    write_register(BMI160_I2C_ADDR, 0x7E, 0x10)  # ACC_sus_MODE
    write_register(BMI160_I2C_ADDR, 0x7E, 0x14)  # GYR_sus_MODE
#    sensor_status = read_register(BMI160_I2C_ADDR, 0x03, 1)  # Read accel data

    # put sensor to sleep
    time_left = time_left - 0.2
    time.sleep(time_left)
    write_register(BMI160_I2C_ADDR, 0x7E, 0x11)  # ACC_NORMAL_MODE
    time.sleep(0.1)
    write_register(BMI160_I2C_ADDR, 0x7E, 0x15)  # GYR_NORMAL_MODE
    time.sleep(0.1)
    return
# Perform auto-calibration
ax_offset, ay_offset, az_offset = auto_calibrate()

num_windows = 2
window_length = 250
window_time = 5
sample_time = 0.02
for w in range(0, num_windows):
    print(w)
    sensor_status = read_register(BMI160_I2C_ADDR, 0x03, 1)  # Read accel data
    ax_arr = np.zeros((window_length, 1))
    gx_arr = np.zeros((window_length, 1))
    ay_arr = np.zeros((window_length, 1))
    gy_arr = np.zeros((window_length, 1))
    az_arr = np.zeros((window_length, 1))
    gz_arr = np.zeros((window_length, 1))

    random_num = np.random.rand()
    random_num=1
    if (0<random_num <= 0.8):
        sample_off = int(2/5*window_length)
    elif (0.8<random_num < 0.9):
        sample_off = int(3/5*window_length)
    else:
        sample_off = int(5/5*window_length)

    for s in range(0, window_length):

        if (s >= sample_off):

            time_remaining = window_time - sample_time*s
            sensor_sleep(time_remaining)
            break
        try:
            # Read acceleration values
            ax, ay, az = read_acceleration(ax_offset, ay_offset, az_offset)
            gx, gy, gz = read_gyroscope(ax_offset, ay_offset, az_offset)

            # Calculate tilt angles
            pitch, roll = calculate_tilt_angles(ax, ay, az)

            ax_arr[s] = ax
            gx_arr[s] = gx
            ay_arr[s] = ay
            gy_arr[s] = gy
            az_arr[s] = az
            gz_arr[s] = gz

        except OSError as e:
            print("I2C Error: ", e)

        time.sleep(sample_time)

    avg_x = np.mean(ax_arr)
    avg_gx = np.mean(gx_arr)
    avg_y = np.mean(ay_arr)
    avg_gy = np.mean(gy_arr)
    avg_z = np.mean(az_arr)
    avg_gz = np.mean(gz_arr)
os.system("pkill -f 'python3 data_logger.py'")
