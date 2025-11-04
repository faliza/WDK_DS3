import pandas as pd
import numpy as np
import os
import sys

# ------ PATH SETUP FIRST ------
DS3_PATH = "/local/data/falizadehziri_l/DS3" # must change
sys.path.append(DS3_PATH)

import common
from DASH_Sim_v0 import run_simulator  # Direct DS3 call

# Load Excel ONCE (low-fidelity metadata source)
excel_file_path = os.path.join(DS3_PATH, "data/Library_detailed_tags_2.xlsx") # must change
df_metrics = pd.read_excel(excel_file_path)  # must include Power + Latency


def Energy(x, fidelity=0, verbose=False):
    x = np.array(x).astype(bool)

    if fidelity == 0:
        selected = df_metrics[x]
        if selected.empty:
            return 0.0
        return float(np.sum(selected["Power_W"] * selected["Latency(for task filtering),micros"]))

    elif fidelity == 1:
        common.selection_vector = x.tolist()
        sim_latency, sim_energy = run_simulator()
        return float(sim_energy)


def Latency(x, fidelity=0, verbose=False):
    x = np.array(x).astype(bool)

    if fidelity == 0:
        selected = df_metrics[x]
        if selected.empty:
            return 0.0
        return  float(np.sum(selected["Latency(for task filtering),micros"]))

    elif fidelity == 1:
        common.selection_vector = x.tolist()
        sim_latency, sim_energy = run_simulator()
        return float(sim_latency)


# -- Test High Fidelity (DS3) Evaluation --
x_test = np.zeros(75, dtype=int)
x_test[2] = 1
x_test[45] = 1
x_test[71] = 1

lat = Latency(x_test, fidelity=1, verbose=True)
eng = Energy(x_test, fidelity=1, verbose=True)

print("DS3 High-Fidelity Simulation Results:")
print("Latency microS:", lat)
print("Energy microJ:", eng)
 