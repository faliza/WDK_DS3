from ina219 import INA219
import sys
import time
import csv
SHUNT_OHMS = 0.1
ina = INA219(SHUNT_OHMS, busnum=1)
ina.configure()

header = ["time s", "power mW"]
filename = "Raspi5_DWT_sh.csv"
data_all = []

#while True:
 #       p = ina.power()
 #       t = time.time()
  #      print(p)
        #print(ina.voltage())
        #data = []
        #data.append(t)
        #data.append(p)
while True:
    voltage = ina.voltage()      # Volts
    current = ina.current()      # mA
    power = ina.power()          # mW

    print(f"V = {voltage:.3f} V | I = {current:.2f} mA | P = {power:.2f} mW")
    time.sleep(1)
