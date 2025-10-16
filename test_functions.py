
import os
import pathlib
from abc import abstractmethod
import math
import numpy as np
import torch
from sklearn.svm import SVR
from torch import Tensor
from xgboost import XGBRegressor
import pandas as pd
from copy import deepcopy
import sys
import subprocess

class TestFunction:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = "categorical"

    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        self.n_vertices = None
        self.config = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self):
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, (
            "int_constrained_dims must be a subset of the continuous_dims, " "but continuous_dims is not supplied!"
        )
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), (
            "all continuous dimensions with integer "
            "constraint must be themselves contained in the "
            "continuous_dimensions!"
        )

    @abstractmethod
    def compute(self, x, normalize=None):
        raise NotImplementedError()

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for _ in range(size):
            x = np.array([np.random.choice(self.config[_]) for _ in range(self.dim)])
            y.append(self.compute(x, normalize=False,))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)
    
#--------------------ComponentSelection---------------------------------

class ComponentSelection(TestFunction):
    def __init__(self, excel_file_path,L1,L2, **tkwargs):
        super().__init__()

        # Load data from Excel
        try:
            df = pd.read_excel(excel_file_path)
            print("Excel file loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Excel file not found at {excel_file_path}. Please check the file name and path.")
            exit()

        # Extract component information
        self.component_names = df["name"].tolist()
        self.component_types = df["type (p=0,s=1,com=2)"].tolist()
        self.component_spaces = df["Space"].tolist()
        self.component_energy = df["Energy"].tolist()
        
        # Shuffle the data by creating a index
        indices = torch.randperm(len(self.component_names))
        #Reassign the old values for new, now reordered
        self.component_names = [self.component_names[i] for i in indices]
        self.component_types = [self.component_types[i] for i in indices]
        self.component_spaces = [self.component_spaces[i] for i in indices]
        self.component_energy = [self.component_energy[i] for i in indices]

        # Convert to tensors
        self.component_types = torch.tensor(self.component_types)
        self.component_spaces = torch.tensor(self.component_spaces, dtype=torch.float64)
        self.component_energy = torch.tensor(self.component_energy, dtype=torch.float64)

        self.n_components = len(self.component_names)
        self.dim = self.n_components
        self.binary_inds = list(range(self.n_components))
        self.bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).to(**tkwargs)
        self.excel_file_path = excel_file_path
        
        self.n_categorical = 0
        self.n_continuous = 0
        self.L1=L1
        self.L2=L2;
        #
    def check_component_constraint(self, x):
        x = torch.as_tensor(x, dtype=torch.bool)
        # print("x:")
        # print(x)
        # print(x.shape)
        
        
        selected_types = self.component_types[x]
        
        # print(f"selected_types: {selected_types}")
        
        
        has_sensor = (1 in selected_types)
        has_communicator = (2 in selected_types)
        has_processor = (0 in selected_types)
        return has_sensor and has_communicator and has_processor

    # def objective_function_space(self, x, constraint_satisfied):
    #     if not constraint_satisfied:
    #         print("------------Penalty-------------")
    #         return torch.tensor(1e9, dtype=torch.float64)  # Return a large penalty if constraint is violated

    #     x = torch.as_tensor(x, dtype=torch.bool)
    #     selected_spaces = self.component_spaces[x]
    #     total_space = torch.sum(selected_spaces)
    #     print("total_space:")
    #     print(total_space)
    #     return total_space
    def objective_function_space(self, x, constraint_satisfied):
        if not constraint_satisfied:
            print("------------Penalty-------------")
            return torch.tensor(1e9, dtype=torch.float64)  # Return a large penalty if constraint is violated
    
        x = torch.as_tensor(x, dtype=torch.bool)
        selected_spaces = self.component_spaces[x]
        selected_energy = self.component_energy[x]
        total_space = torch.sum(selected_spaces)
        total_energy = torch.sum(selected_energy)
        
        # Calculate the penalty based on the number of selected components
        P =0  # Define a constant penalty value. Test and set good numbers!
        num_selected_components = torch.sum(x.int())  # Count the selected components (True values)
        penalty = num_selected_components * P # The goal
        L1=self.L1
        L2=self.L2
        # Add the penalty to the total space
        objective_value = (L1*total_space + L2*total_energy+ penalty)
        print(f"-------------objective_value:{objective_value}-----------") 
        print(f"total_space :{total_space}, total_energy :{total_energy} Penalty :{penalty}") #Check the final number
        
        
        
        return objective_value

    def compute(self, x, normalize=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x_bool = x.to(torch.bool)  # Convert to boolean for the constraint
        constraint_satisfied = self.check_component_constraint(x_bool)
        objective_value = self.objective_function_space(x_bool, constraint_satisfied)
        return objective_value

#--------------------ComponentSelection end---------------------------------
# %%


#--------------------FlexibleComponentSelector --------------------------------------

class FlexibleComponentSelector(TestFunction):
    def __init__(self, excel_file_path, L1, L2, required_counts=None, **tkwargs):
        super().__init__()

        
        try:
            df = pd.read_excel(excel_file_path)
            print("Excel file loaded successfully.")
        except FileNotFoundError:
            print(f" Excel file not found at {excel_file_path}")
            exit()

        
        self.component_names = df["name"].tolist()
        self.component_spaces = torch.tensor(df["Space"].tolist(), dtype=torch.float64)
        self.component_energy = torch.tensor(df["Energy"].tolist(), dtype=torch.float64)

        # --- Multi-hot encode the 'type' column ---
        types_raw = df["type"].astype(str).tolist()
        split_tags = [t.replace(" ", "").split(",") for t in types_raw]  # remove whitespace

        all_tags = sorted(set(tag for tags in split_tags for tag in tags))
        self.tag_names = all_tags
        self.tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}

        num_components = len(split_tags)
        num_tags = len(all_tags)
        type_features = torch.zeros((num_components, num_tags), dtype=torch.float64)
        for i, tags in enumerate(split_tags):
            for tag in tags:
                if tag in self.tag_to_idx:
                    type_features[i, self.tag_to_idx[tag]] = 1.0
        self.type_features = type_features
        self.type_features = self.type_features.to(**tkwargs)

        # Optimization setup
        self.n_components = len(self.component_names)
        self.dim = self.n_components
        self.binary_inds = list(range(self.n_components))
        self.bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).to(**tkwargs)

        self.n_categorical = 0
        self.n_continuous = 0
        self.excel_file_path = excel_file_path

        self.L1 = L1
        self.L2 = L2
        self.required_counts = required_counts  # constraint: required number of tags

        # Precompute total tag counts in library
        self.tag_counts = self.type_features.sum(dim=0).int()
        # print("\nTag counts in component library:")
        # for tag, count in zip(self.tag_names, self.tag_counts):
        #     print(f"{tag:10s}: {count.item()}")

    def check_component_constraint(self, x: torch.Tensor, required_counts: torch.Tensor) -> bool:
        x = x.to(torch.bool)
        tag_counts = self.type_features[x].sum(dim=0).int()

        print("\nTag constraint check:")
        for i, required in enumerate(required_counts):
            if required > 0:
                print(f"{self.tag_names[i]:10s}: required {required}, selected {tag_counts[i].item()}")

        return torch.all(tag_counts >= required_counts).item()

    def objective_function(self, x: torch.Tensor, required_counts: torch.Tensor = None) -> torch.Tensor:
        if required_counts is None:
            if self.required_counts is None:
                raise ValueError("No constraint vector provided.")
            required_counts = self.required_counts
    
        x = x.to(torch.bool)
    
        # === Pre-check debug info ===
        num_selected = x.sum().item()
        selected_names = [self.component_names[i] for i, val in enumerate(x) if val]
        type_summary = self.type_features[x].sum(dim=0).int()
        selected_tags = [self.tag_names[i] for i, val in enumerate(type_summary) if val > 0]
        selected_tag_counts = {
            self.tag_names[i]: type_summary[i].item()
            for i in range(len(self.tag_names))
        }
    
        # print("\nSelected Tag Counts")
        # for tag, count in selected_tag_counts.items():
        #     print(f"{tag:10s}: {count}")
    
        print("\nEvaluation Summary:")
        # print(f"Selected components ({num_selected}): {selected_names}")
        print(f"Number of Selected components ({num_selected})")
        # print(f"Selected tags: {selected_tags}")
    
        # === Constraint check ===
        if not self.check_component_constraint(x, required_counts):
            print("Tag constraints violated. Returning penalty.")
            return torch.tensor(1e9, dtype=torch.float64)
    
        # === Objective computation ===
        p=0;
        total_space = self.component_spaces[x].sum()
        total_energy = self.component_energy[x].sum()
        objective_value = self.L1 * total_space + self.L2 * total_energy+p
    
        print(f"Total space: {total_space.item():.2f}, Total energy: {total_energy.item():.2f}")
        print(f"Objective = L1*space + L2*energy + p = {self.L1}*{total_space:.2f} + {self.L2}*{total_energy:.2f} + {p}")
        print(f"Final objective: {objective_value:.4f}")
    
        return objective_value


    def compute(self, x, normalize=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(**tkwargs)
        x_bool = x.to(torch.bool)
        

        return self.objective_function(x_bool)

# %% ------- with tasks-----------


class ComplexDifferent(TestFunction):
    def _normalize_tensor(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())
    def __init__(self, excel_file_path, L1, L2,L3, verbose=False, **tkwargs):
        super().__init__()

        self.verbose = verbose  # Added verbose flag

        try:
            df = pd.read_excel(excel_file_path)
            if self.verbose:
                print("Excel file loaded successfully.")
        except FileNotFoundError:
            print(f" Excel file not found at {excel_file_path}")
            # exit()
            sys.exit()

        self.component_names = df["name"].tolist()
        self.component_spaces = torch.tensor(df["Space"].tolist(), dtype=torch.float64)
        self.component_energy = torch.tensor(df["Energy"].tolist(), dtype=torch.float64)
        # this is not latency it's Throughput (Mbps) but I need to change like 10 variable names so let it be
        self.component_latency = torch.tensor(df["Throughput"].tolist(), dtype=torch.float64)

        # In ComplexDifferent.__init__()
        self.component_spaces = self._normalize_tensor(self.component_spaces)
        self.component_energy = self._normalize_tensor(self.component_energy)
        self.component_latency = self._normalize_tensor(self.component_latency)
        # self.P = P
        



        # --- Multi-hot encode the 'type' column ---
        types_raw = df["type"].astype(str).tolist()
        split_tags = [t.replace(" ", "").split(",") for t in types_raw]  # remove whitespace

        all_tags = sorted(set(tag for tags in split_tags for tag in tags))
        # print(f"--------------------all_tags : {all_tags}")
        self.tag_names = all_tags
        self.tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}

        num_components = len(split_tags)
        num_tags = len(all_tags)
        type_features = torch.zeros((num_components, num_tags), dtype=torch.float64)
        for i, tags in enumerate(split_tags):
            for tag in tags:
                if tag in self.tag_to_idx:
                    type_features[i, self.tag_to_idx[tag]] = 1.0
        self.type_features = type_features

        # Optimization setup
        self.n_components = len(self.component_names)
        self.dim = self.n_components
        self.binary_inds = list(range(self.n_components))
        self.bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).to(**tkwargs)

        self.n_categorical = 0
        self.n_continuous = 0
        self.excel_file_path = excel_file_path

        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        App_Addr = "/local/data/falizadehziri_l/DS3/data/complex_difference_2.app" 
        # App_Addr = r"C:\Users\yuki\Desktop\PhD\Research\complex_different.app" 
        self.task_tags_matrix = self.build_task_tags_matrix_from_app(App_Addr)


        # self.required_counts = required_counts  # constraint: required number of tags

        # Precompute total tag counts in library
        self.tag_counts = self.type_features.sum(dim=0).int()
        # if self.verbose:
        #     print("\nTag counts in component library:")
        #     for tag, count in zip(self.tag_names, self.tag_counts):
        #         print(f"{tag:10s}: {count.item()}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type_features = self.type_features.to(device)
        self.component_spaces = self.component_spaces.to(device)
        self.component_energy = self.component_energy.to(device)
        self.component_latency = self.component_latency.to(device)
        self.task_tags_matrix = self.task_tags_matrix.to(device)
        self.bounds = self.bounds.to(device)

    # ---------------------- Multi-Component Aggregation (MCA)
    # def check_component_constraint(self, x: torch.Tensor) -> bool:
    #     x = x.to(torch.bool)

    #     # Count how many times each tag appears in selected components
    #     # tag_counts = self.type_features[x].sum(dim=0).int()

    #     # Load task-to-tag matrix
    #     task_tags = self.task_tags_matrix

    #     num_tasks = task_tags.shape[1]
    #     num_tags = task_tags.shape[0]
    #     penalty = False
        

    #     # Check if any task requires a tag that is unavailable
    #     # for j in range(num_tasks):  # for each task
    #     #     for i in range(num_tags):  # for each tag
    #     #         if task_tags[i, j] > tag_counts[i]:
    #     #             penalty = True
    #     #             if self.verbose:
    #     #                 print(f"Task {j} requires tag '{self.tag_names[i]}' which is unavailable.")

    #     # return not penalty
        
    #     selected_tag_presence = self.type_features[x].any(dim=0)  # shape: (num_tags,)
        
    #     missing_tag_mask = (task_tags.T @ (~selected_tag_presence).float()) > 0

    #     if missing_tag_mask.any():
    #         if self.verbose:
    #             missing_tasks = torch.nonzero(missing_tag_mask).squeeze().tolist()
    #             if not isinstance(missing_tasks, list):
    #                 missing_tasks = [missing_tasks]
    #             for t in missing_tasks:
    #                 print(f"[BO]  Task {t} requires at least one tag that is missing in selected components.")
    #         return False

    #     return True

    #----------------------Single-Component Satisfaction (SCS)
    def check_component_constraint(self, x: torch.Tensor) -> bool:
        x = x.to(torch.bool)
        task_tags = self.task_tags_matrix  # shape: [num_tags, num_tasks]

        selected_type_features = self.type_features[x]  # shape: [#selected_comps, #tags]
        num_tasks = task_tags.shape[1]

        for j in range(num_tasks):  # For each task
            required_tag_indices = (task_tags[:, j] == 1).nonzero(as_tuple=True)[0]

            task_satisfied = False
            for comp_idx in range(selected_type_features.shape[0]):
                comp_tags = selected_type_features[comp_idx]
                if torch.all(comp_tags[required_tag_indices] == 1):
                    task_satisfied = True
                    break  # Found a suitable component for this task

            if not task_satisfied:
                if self.verbose:
                    missing_tags = [self.tag_names[i] for i in required_tag_indices.tolist()]
                    print(f"[BO]  Task {j} requires tags {missing_tags}, but no selected component has all of them.")
                return False  # This task cannot be satisfied

        return True  # All tasks satisfied
    
    def objective_function(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.type_features.device)
        x = x.to(torch.bool)
    
        # === Pre-check debug info ===
        num_selected = x.sum().item()
    
        tag_counts = self.type_features[x].sum(dim=0).int()
        selected_tags = [self.tag_names[i] for i, val in enumerate(tag_counts) if val > 0]
    
        selected_tag_counts = {
            self.tag_names[i]: tag_counts[i].item()
            for i in range(len(self.tag_names))
        }
    
        if self.verbose:
            print("\nSelected Tag Counts")
            for tag, count in selected_tag_counts.items():
                print(f"{tag:10s}: {count}")
    
            print("\nEvaluation Summary:")
            print(f"Selected components ({num_selected})")
            print(f"Selected tags: {selected_tags}")
    
        # === Constraint check ===
        if not self.check_component_constraint(x):
            if self.verbose:
                print("Tag constraints violated. Returning penalty.")
            return torch.tensor(1e9, dtype=torch.float64)
    
        else:
            # === Objective computation ===
            total_space = self.component_spaces[x].sum()
            total_energy = self.component_energy[x].sum()
            total_latency = self.evaluate_total_latency(x)  # get latency from the method
    
            objective_value = (
                self.L1 * total_space +
                self.L2 * total_energy -
                self.L3 * total_latency
            )
    
            if self.verbose:
                print(f"Total space: {total_space.item():.2f}, Total energy: {total_energy.item():.2f}, Total latency: {total_latency:.2f}")
                print(f"Objective = {self.L1}*space + {self.L2}*energy - {self.L3}*Throughput")
                print(f"Final objective: {objective_value:.4f}")
    
            return objective_value


    def compute(self, x, normalize=None):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x_bool = x.to(torch.bool)

        return self.objective_function(x_bool)

    def build_task_tags_matrix_from_app(self, app_file_path):

        # Define mapping from keywords in .app to known tags
        keyword_to_tags = {
            "accelerometer": ["ACC"],
            "gyroscope": ["GYRO"],
            "temperature": ["TEMP"],
            "EMG": ["EMG"],
            "EEG": [],
            "orientation": [],
            "calibration": [],
            "filtering": [],
            "cpu": ["MCU"],
            "dsp": ["MCU"],
            "zigbee": ["ZIGBEE"],
            "antenna": ["TRANSCEIVER"],
            "blu": ["BT"]
        }

        # Parse .app file
        task_types = {}
        with open(app_file_path, 'r') as f:
            lines = f.readlines()

        current_task_id = None
        for line in lines:
            line = line.strip()
            if line.startswith("ID"):
                current_task_id = int(line.split()[1])
                task_types[current_task_id] = []
            elif line.startswith("type"):
                keywords = line.split()[2:]  # skip "type" and task category
                task_types[current_task_id].extend(keywords)

        task_ids = sorted(task_types.keys())
        num_tags = len(self.tag_names)
        num_tasks = len(task_ids)
        task_tags_matrix = torch.zeros((num_tags, num_tasks), dtype=torch.float32)

        # Fill matrix
        for j, task_id in enumerate(task_ids):
            for keyword in task_types[task_id]:
                for tag in keyword_to_tags.get(keyword, []):
                    if tag in self.tag_names:
                        i = self.tag_names.index(tag)
                        task_tags_matrix[i, j] = 1.0

        if self.verbose:
            print("\nTask-Tags matrix created. Shape: ", task_tags_matrix.shape)
            # print("\nTask-Tags matrix created :", task_tags_matrix)
        return task_tags_matrix


# excel_path = r"C:\Users\f.alizadehziri\Desktop\PHD\Term 1\Research\Library_ detailed_tags.xlsx"    
# app = task1(excel_path, L1=1.0, L2=1.0)
# matrix = app.build_task_tags_matrix_from_app("complex_different.app")

    def evaluate_total_latency(self, x: torch.Tensor) -> float:
        """
        Computes total latency score: for each task, sum the worst latency per required tag,
        then sum this value across all tasks.
        """
        x = x.to(torch.bool)
        selected_type_features = self.type_features[x]         # [#selected_components x #tags]
        selected_latencies = self.component_latency[x]         # [#selected_components]
        print(f"selected_type_features{selected_type_features}")
        print(f"selected_type_features shape{selected_type_features.shape}")
        # Get task_tags matrix
        task_tags = self.task_tags_matrix  # [num_tags x num_tasks]
        num_tags, num_tasks = task_tags.shape
    
        total_latency = 0.0
    
        for j in range(num_tasks):  # For each task
            print(f"Task {j}:")
            task_latency = 0.0
    
            for i in range(num_tags):  # For each tag
                if task_tags[i, j] == 1:
                    print(f"task tag{ i} is selcted for task {j} ")
                    # Find components that have this tag
                    tag_mask = selected_type_features[:, i] > 0
                    print(f"tag_mask{tag_mask}")
                    # print(f"component {i} has  ")
                    if tag_mask.any():
                        print(f" selected_latencies{selected_latencies[tag_mask]}")
                        # print(f"selected_latencies[tag_mask]{selected_latencies[tag_mask].item}")
                        worst_latency = selected_latencies[tag_mask].min().item()
                        task_latency += worst_latency
    
            total_latency += task_latency
    
        if self.verbose:
            print(f"\nTotal latency across all tasks: {total_latency:.4f}")
    
        return total_latency


# %%

class DS3DrivenObjective(ComplexDifferent):
    """
    A subclass of ComplexDifferent that overrides the objective evaluation by invoking
    DS3 to simulate task scheduling and execution. It sends `x`, `type_features`, and `task_tags_matrix`
    to DS3, retrieves space, energy, and latency, then combines them using L1, L2, L3.
    """
    def __init__(self, excel_file_path: str, selection_output_file: str, result_input_file: str, L1: float = 0.0, L2: float = 0.0, L3: float = 1.0, P: int=0, verbose=False, **tkwargs):
        super().__init__(
            excel_file_path=excel_file_path,
            L1=L1,
            L2=L2,
            L3=L3,
            # P=P,
            verbose=verbose,
            **tkwargs
        )
        device = self.type_features.device
        self.selection_output_file = selection_output_file
        self.result_input_file = result_input_file
        self.execution_times = []
        self.throughputs = []

    def call_ds3_simulator(self, x: torch.Tensor) -> tuple:
        if self.verbose:
            print("\n[BO]  Starting DS3 simulator call...")
        """
        Sends component selection `x`, tag matrix, and task tags to DS3.
        DS3 returns:
            - latency (average job latency)
            - execution_time (total simulation time in µs)
            - cumulative_time (sum of job runtimes)
            - energy (Joules)
            - edp (Energy × Delay Product)
            - avg_concurrent_jobs
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.type_features = self.type_features.to(device)
        x_np = x.cpu().numpy().astype(int)
        type_features_np = self.type_features.cpu().numpy()
        task_tags_np = self.task_tags_matrix.cpu().numpy()
    
        # Save input data to .npz
        if self.verbose:
            print("Number of selected components (BO):", sum(x_np))
            print("[BO → DS3]  Saving input .npz for DS3...")
        np.savez(self.selection_output_file, x=x_np, type_features=type_features_np, task_tags=task_tags_np)
    
        # Run DS3 via shell script
        # os.system("./run_ds3_simulation.sh")
        if self.verbose:
            print("[BO → DS3]  Launching DS3 simulation...")
        try:
            subprocess.run(
                ["python", "DASH_Sim_v0.py"],
                cwd="/local/data/falizadehziri_l/DS3",
                check=True
                # stdout=sys.stdout,
                # stderr=sys.stderr
            )
        except subprocess.CalledProcessError as e:
            print("DS3 simulation failed:", e)
            return 1e9  # Penalty for failure
        
        # ---------------------------
        if self.verbose:
            print("[DS3 → BO]  Reading simulation results from:", self.result_input_file)
        try:
            with open(self.result_input_file, 'r') as f:
                lines = f.readlines()
                execution_time = float(lines[0].strip())         # First line
                total_throughput = float(lines[1].strip())       # Second line

                
                # lines = f.readlines()
                # latency = float(lines[0].strip())             # Ave latency
                # execution_time = float(lines[1].strip())      # Execution time(us)
                # cumulative_time = float(lines[2].strip())     # Cumulative Execution time(us)
                # energy = float(lines[3].strip())              # Total energy consumption(J)
                # edp = float(lines[4].strip())                 # EDP
                # avg_concurrent_jobs = float(lines[5].strip()) # Average concurrent jobs
    
            # return latency, execution_time, cumulative_time, energy, edp, avg_concurrent_jobs
            print(f"execution_time: {execution_time} ")
            print(f"total_throughput: {total_throughput} ")
            return execution_time, total_throughput

        except Exception as e:
            print("Failed to read DS3 result:", e)
            return 1e9


    def compute(self, x: torch.Tensor, normalize=None) -> torch.Tensor:
        # if self.verbose:
        #     print("compute")
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(torch.bool)
        numberOfSeklecion=sum(x)
        # print(x)
        P=1;
        if not self.check_component_constraint(x):
            if self.verbose:
                print("Tag constraints violated. Returning penalty 1e9.")
            self.execution_times.append(1e9)
            self.throughputs.append(1e9)
            return torch.tensor(1e9, dtype=torch.float64)
    
        # Unpack full DS3 metrics
        execution_time,total_throughput= self.call_ds3_simulator(x)
        self.execution_times.append(execution_time)
        self.throughputs.append(total_throughput)
        penalty=sum(x)*P
        print(f"Penalty :{sum(x)} * {P} = {penalty}")
        # Example objective function: L1 * energy - L2 * latency + L3 * execution_time
        # objective_value = self.L1 * energy - self.L2 * latency + self.L3 * execution_time
        objective_value = penalty+self.L2*1/(total_throughput)+ self.L3 * execution_time
    
        if self.verbose:
            
            # print(f"DS3 returned - Latency: {latency:.4f}, Energy: {energy:.4f}, Execution Time: {execution_time:.4f}, "
            #       f"Cumulative Time: {cumulative_time:.4f}, EDP: {edp:.4f}, Avg Concurrent Jobs: {avg_concurrent_jobs:.4f}")
            # print(f"Objective = {self.L1}*energy - {self.L2}*latency + {self.L3}*execution_time")
            print(f"Final objective: {objective_value:.4f}")
    
        return torch.tensor(objective_value, dtype=torch.float64)



# %% dummy run


if __name__ == "__main__":
    import torch
    from pprint import pprint

    # === Config ===
    excel_path = "/local/data/falizadehziri_l/DS3/data/Library_detailed_tags2.xlsx"  
    L1, L2, L3 = 0.2, 0.8, 1.0  # space, energy, latency weights
    tkwargs = {"dtype": torch.double, "device": "cpu"}

    # === Create instance ===
    f = ComplexDifferent(excel_file_path=excel_path, L1=L1, L2=L2, L3=L3, verbose=True, **tkwargs)

    # === Show tag list ===
    print("\nAvailable Tags:")
    for i, tag in enumerate(f.tag_names):
        print(f"{i:2d} - {tag}")

    # === Pick a test selection vector ===
    x_test = torch.zeros(f.dim)
    # Select 3 components manually (adjust the indices to your Excel file size)
    x_test[[0,1, 2,3,4, 10,32,33,34,35,43,61,62,81,82,94]] = 1  

    # === Run constraint check and compute ===
    print("\nChecking tag constraints for selection...")
    if f.check_component_constraint(x_test):
        print(" Constraint satisfied. Computing objective...")
        obj = f.objective_function(x_test)
        print(f"Objective value: {obj:.4f}")

        # latency = f.evaluate_total_latency(x_test)
        # print(f"Total latency: {latency:.4f}")
    else:
        print(" Constraint violated.")


