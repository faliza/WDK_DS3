# This code implements a framework for running BO experiemnts designed for evaluating black-box functions 
# under different configurations. It supports multiple evaluation functions such as LABS, MaxSAT60, Ackley53, 
# and SVM, each with specific constraints on binary, categorical, and continuous parameters.

# How to modify the code so it fits our needs?
#%%
import inspect  # For retrieving information about live objects like functions and their arguments
import pickle    # For serializing and deserializing objects to save experiment data
import time     # Measuring execution time
import warnings     # To suppress warnings during execution
from copy import deepcopy   # For creating deep copies of complex objects
from typing import Dict, Optional, Union #For better code readability

import torch    # For tensor computations
from torch import Tensor
import numpy as np    
from torch.quasirandom import SobolEngine   # For generating quasi-random numbers

#from  import fit_gpytorch_model // error 1: cannot import name 'fit_gpytorch_model' from 'botorch' - suggests fit_gpytorch_mll instead
from botorch import fit_gpytorch_mll #error 2: ImportError: attempted relative import with no known parent package
import pandas as pd  
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
# Import optimization and modeling utilities from BoTorch and GPyTorch libraries
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.objective import WeightedMCMultiOutputObjective

from botorch.models import SingleTaskGP
from botorch.models.deterministic import GenericDeterministicModel
from botorch.models.fully_bayesian import MIN_INFERRED_NOISE_LEVEL
from botorch.models.model import ModelList
from botorch.models.transforms import Normalize, Standardize
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.utils.warnings import NumericalWarning
from botorch.acquisition.analytic import LogExpectedImprovement 

from dictionary_kernel import DictionaryKernel # Error 3: ImportError: attempted relative import with no known parent package
from optimize import optimize_acq_function_mixed_alternating, optimize_acqf_binary_local_search

from test_functions import ComponentSelection 
from test_functions import FlexibleComponentSelector
from test_functions import ComplexDifferent
from test_functions import DS3DrivenObjective



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {"device": device, "dtype": torch.float32}  # use as default



# This function arranges multiple independent runs of a Bayesian optimization experiment 
# and returns experiment data (Xs, Ys, metadata) and optionally saves them to a file
def run_experiment(
    n_replications: int,  # number of independent experiment repetitions
    base_seed: int = 1234,  # base seed for reproducibility across repetitions
    save_to_pickle: bool = True,  # whether to save results to a pickle file
    fname: Optional[str] = None,  # optional custom filename for saving results
    
    excel_file_path: str = "/local/data/falizadehziri_l/DS3/data/Library_detailed_tags_2.xlsx",
    
    # excel_file_path: str = r"C:\Users\yuki\Desktop\PhD\Research\Library_ detailed_tags.xlsx",
    
    
    required_counts: Optional[torch.Tensor] = None,  # 
    **kwargs,  # additional experiment parameters passed as a dictionary
) -> Dict[str, Union[str, Optional[Tensor]]]:

    Xs, Ys, metadata = [], [], []  # lists to collect input points, outcomes, and metadata for all runs

    for i in range(n_replications):  # loop over the number of replications

        print(f"=== Replication {i + 1}/{n_replications} ===")  # progress tracking

        
        X, Y, meta, f = _run_single_trial(
            torch_seed=base_seed + i,
            excel_file_path=excel_file_path,
            required_counts=required_counts,  # 
            **kwargs
        )

        if save_to_pickle:  # save intermediate results if required

            # constructing a default filename if none is provided
            fname = (
                fname
                or "./"
                + "ComponentSelection"
                + "_n0=" + str(kwargs["n_initial_points"])
                + "_n=" + str(kwargs["max_evals"])
                + "_q=" + str(kwargs["batch_size"])
                + ".pkl"
            )
            pickle.dump((Xs, Ys, metadata), open(fname, "wb"))  # save data using pickle
            print(f"Results saved to: {fname}")  # printing save location

        Xs.append(X)    # collect input points from this trial
        Ys.append(Y)    # collect outcomes from this trial
        metadata.append(meta)  # collect trial-specific metadata

    Xs, Ys = torch.stack(Xs), torch.stack(Ys)  # combine data from all trials into tensors

    if save_to_pickle:  # repeat saving after all trials
        pickle.dump((Xs, Ys, metadata), open(fname, "wb"))
        print(f"Results saved to: {fname}")

    return Xs, Ys, metadata, f  # return the collected data

def _run_single_trial(
    torch_seed: int,  # seed for reproducibility
    evalfn: str,  # name of the evaluation function 
    max_evals: int,  # max num of function evaluations
    n_initial_points: int,  # num of initial samples for the optimization
    batch_size: int = 1,  # num of points to evaluate in each iteration
    n_binary: int = 0,  # num of binary parameters in the search space
    n_categorical: int = 0,  # num of categorical parameters
    n_continuous: int = 0,  # num of continuous parameters
    init_with_k_spaced_binary_sobol: bool = True,  # custom initialization 
    n_prototype_vectors: int = 10,  # num of prototype vectors for the custom kernel
    verbose: bool = False,  # verbose mode - helps debugging
    df=None,
    feature_costs: Optional[Tensor] = None,  # feature costs for multi-objective evaluation
    
    L1: float = 1,  # **Added L1**
    L2: float = 1,  # **Added L2**
    L3: float = 1,  # **Added L2**
    # P : int= 0,
    
    # excel_file_path: str = r"C:\Users\yuki\Desktop\PhD\Research\Library_ detailed_tags.xlsx" , 
    excel_file_path: str = "/local/data/falizadehziri_l/DS3/data/Library_ detailed_tags.xlsx",
    required_counts: Optional[torch.Tensor] = None,
) -> Dict[str, Union[str, Optional[Tensor]]]:
    
    _run_single_trial_input_kwargs = deepcopy(inspect.getargvalues(inspect.currentframe())[-1])
    start_time = time.time()
    #------------------For reproducibility----------------
    torch.manual_seed(torch_seed) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # use GPU if available
    tkwargs = {"dtype": torch.double, "device": device} # tensor options for computation
    
    if verbose:
        print(tkwargs)
    
    if evalfn == "ComponentSelection":
        f = ComponentSelection(excel_file_path=excel_file_path, L1=L1, L2=L2, **tkwargs)
        n_binary = f.dim
        assert n_categorical == 0, "ComponentSelection has no categorical parameters"
        assert n_continuous == 0, "ComponentSelection has no continuous parameters"
        
    elif evalfn == "FlexibleComponentSelector":
        assert df is not None, "Excel DataFrame must be provided for ComplexDifferent."
        assert required_counts is not None, "You must provide `required_counts` when using ComplexDifferent."
    
        f = FlexibleComponentSelector(
            excel_file_path=excel_file_path,
            L1=L1,
            L2=L2,
            required_counts=required_counts,
            **tkwargs
        )
        n_binary = f.dim
        assert n_categorical == 0
        assert n_continuous == 0
        
    elif evalfn == "ComplexDifferent":
        assert df is not None, "Excel DataFrame must be provided for ComplexDifferent."
        
        f = ComplexDifferent(
            excel_file_path=excel_file_path,
            L1=L1,
            L2=L2,
            L3=L3,                
            verbose=verbose,      
            **tkwargs
        )
        n_binary = f.dim
        assert n_categorical == 0
        assert n_continuous == 0    
        
    elif evalfn == "DS3DrivenObjective":
        assert df is not None, "Excel DataFrame must be provided for DS3DrivenObjective."
        # assert required_counts is not None, "You must provide `required_counts` when using DS3DrivenObjective."
    
        f = DS3DrivenObjective(
            excel_file_path=excel_file_path,
            selection_output_file="bo_input_vectors.npz",       # <-- Save BO input for DS3 to read
            result_input_file="ds3_sim_results.txt",            # <-- DS3 will write back results here
            L1=L1,
            L2=L2,
            L3=L3,
            # P=P,
            verbose=verbose,
            **tkwargs
        )
        n_binary = f.dim
        assert n_categorical == 0
        assert n_continuous == 0

    else:
        raise ValueError(f"Unknown evalfn {evalfn}")   # handle invalid inputs  

    # Get initial Sobol points
    # X = SobolEngine(dimension=f.dim, scramble=True, seed=torch_seed).draw(n_initial_points).to(**tkwargs)
    X = SobolEngine(dimension=f.dim, scramble=True, seed=torch_seed).draw(n_initial_points)
    X = X.to(dtype=torch.float32, device='cpu')

    if init_with_k_spaced_binary_sobol:
        X[:, f.binary_inds] = 0
        with torch.random.fork_rng():
            #------------------For reproducibility----------------
            torch.manual_seed(torch_seed)
            #-------------------constraint to pick at leas 3
            # k = torch.randint(low=1, high=n_binary - 1, size=(n_initial_points,), device=device)
            k = torch.randint(low=1, high=n_binary - 1, size=(n_initial_points,), device='cpu')

            # binary_inds = torch.tensor(f.binary_inds, device=device)
            binary_inds = torch.tensor(f.binary_inds, dtype=torch.long, device='cpu')

            for i in range(n_initial_points):
                X[i, binary_inds[torch.randperm(n_binary)][: k[i]]] = 1

    # Rescale the Sobol points
    X = f.bounds[0] + (f.bounds[1] - f.bounds[0]) * X
    X[:, f.binary_inds] = X[:, f.binary_inds].round()  # Round binary variables
    assert f.n_categorical == 0, "TODO"
    # Y = torch.tensor([f(x) for x in X]).to(**tkwargs)
    Y = torch.tensor([f(x) for x in X], dtype=torch.float32, device='cpu')

    assert Y.ndim == 2 if evalfn == "SVM" else Y.ndim == 1

    afo_config = {
        "n_initial_candts": 2000,
        "n_restarts": 20,
        "afo_init_design": "random",
        "n_alternate_steps": 50,    
        "num_cmaes_steps": 50,
        "num_ls_steps": 50,
        "n_spray_points": 200,
        "verbose": False,
        "add_spray_points": True,
        "n_binary": n_binary,
        "n_cont": n_continuous,
    }
    o=0;
    while len(X) < max_evals:
        print(f"--------------------------------Iteration : {o}")
        o += 1
        likelihood = GaussianLikelihood(
            noise_prior=GammaPrior(torch.tensor(0.9, **tkwargs), torch.tensor(10.0, **tkwargs)),
            noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL),
        )
        dictionary_kernel = DictionaryKernel(
            num_basis_vectors=n_prototype_vectors,
            binary_dims=f.binary_inds,
            num_dims=f.dim,
            similarity=True,
        )
        covar_module = ScaleKernel(
            base_kernel=dictionary_kernel,
            outputscale_prior=GammaPrior(torch.tensor(2.0, **tkwargs), torch.tensor(0.15, **tkwargs)),
            outputscale_constraint=GreaterThan(1e-6)
        )
        train_Y = (Y - Y.mean()) / Y.std() #if evalfn != "SVM" else (Y[:, 0] - Y[:, 0].mean()) / Y[:, 0].std()
        gp_model = SingleTaskGP(
            train_X=X,
            train_Y=train_Y.unsqueeze(-1),
            covar_module=covar_module,
            input_transform=Normalize(d=X.shape[-1]),
            likelihood=likelihood,
        )
        mll = ExactMarginalLogLikelihood(model=gp_model, likelihood=gp_model.likelihood)
        fit_gpytorch_mll(mll)

        if batch_size == 1:
            acqf = LogExpectedImprovement (model=gp_model, best_f=train_Y.min())
            pareto_points = X[torch.argmin(train_Y)].unsqueeze(0).clone()
        else:
            acqf = LogExpectedImprovement (model=gp_model, best_f=train_Y.min())
            pareto_points = X[torch.argmin(train_Y)].unsqueeze(0).clone()

        with warnings.catch_warnings():  # Filter jitter warnings
            warnings.filterwarnings("ignore", category=NumericalWarning)
            if n_binary > 0 and n_continuous == 0:
                next_x, acq_val = optimize_acqf_binary_local_search(
                    acqf, afo_config=afo_config, pareto_points=pareto_points, q=batch_size
                )
            elif n_binary > 0 and n_continuous > 0:  # mixed search space
                cont_dims = torch.arange(n_binary, n_binary + n_continuous, device=device)
                next_x, acq_val = optimize_acq_function_mixed_alternating(
                    acqf, cont_dims=cont_dims, pareto_points=pareto_points, q=batch_size, afo_config=afo_config,
                )
               
        # print("Proposed Component Selection:")
        # print(next_x)
        # n=0;
        # for i in range(n_binary):
        #     if next_x[0][i]==1:
        #         n=n+1;
            
        # print(f"------------number of chosen components:{n}------------")   
        
        X = torch.cat([X, next_x])
        Y = torch.cat([Y, torch.tensor([f(x) for x in next_x], **tkwargs)])
        # Find the best solution and retrieve components, as I understood from the previous code in your old code base
        
        if verbose:
            
            print(
                    f"best value = {torch.min(Y):.6f}  | at {torch.argmin(Y)}, "
                    # f"best point no of ones @ {torch.sum(X[torch.argmax(Y)])}, "
                    # f"unique x's = {len(torch.unique(X, dim=0))}, "
                    f"acq value = {acq_val.item():.2e}"
                )
            
    print("--------End of iterations--------")        
    best_idx = torch.argmin(Y)
    best_x = X[best_idx]  #This gives the best vector
    best_objective_value = Y[best_idx]
    
    
    #Print the info

     # Convert best_x to boolean (0s and 1s) and get binary_inds
    best_x_binary = (best_x).round().to(torch.bool) #Convert, and not select, since best_x is just one number
    
    # print(f"best objective value {best_objective_value} \nBest Component Selection: {best_x_binary}")
    # print(f"best objective value {best_objective_value} ")
    
    
    selected_components = [f.component_names[i] for i in range(f.n_components) if best_x_binary[i]]
    print("Best Component Selection name:")
    print(selected_components)
    
    # print(f"Objective: {best_objective_value}")
    num_ones = sum(best_x)
    print(f"Number of best components selected: {num_ones}")
    
    
    # selected_components_energy = [sum(f.component_energy[i] for i in range(f.n_components) if best_x_binary[i])]
    # print(f"Real energy: {selected_components_energy}")
    
    # selected_components_space = [sum(f.component_spaces[i] for i in range(f.n_components) if best_x_binary[i])]
    # print(f"Real space: {selected_components_space}")
    print(f"Total obj: {best_objective_value}")
    
   
    X, Y = X[:max_evals], Y[:max_evals]
    # Save the results to manifold
    end_time = time.time()
    metadata = {
        "total_time": end_time - start_time,
        "kwargs": _run_single_trial_input_kwargs,
    }
    
    return X, Y, metadata, f


def make_required_vector(all_tags: list, constraints: dict) -> torch.Tensor:
   
    required = torch.zeros(len(all_tags), dtype=torch.int)
    for tag, count in constraints.items():
        if tag not in all_tags:
            raise ValueError(f"Tag '{tag}' not found in all_tags.")
        idx = all_tags.index(tag)
        required[idx] = count
    return required



excel_file_path = "/local/data/falizadehziri_l/DS3/data/Library_detailed_tags_2.xlsx"
# excel_file_path = r"C:\Users\yuki\Desktop\PhD\Research\Library_ detailed_tags.xlsx"

df = pd.read_excel(excel_file_path)


types_raw = df["type"].astype(str).tolist()
split_tags = [t.replace(" ", "").split(",") for t in types_raw]
all_tags = sorted(set(tag for tags in split_tags for tag in tags))




# # Run experiment
# X, Y, metadata, f = run_experiment(
#     n_replications=1,
#     evalfn="ComplexDifferent",
#     max_evals=80,
#     n_initial_points=30,
#     batch_size=1,
#     # n_binary=96,
#     n_categorical=0,
#     n_continuous=0,
#     init_with_k_spaced_binary_sobol=True,
#     n_prototype_vectors=32,
#     verbose=True,
#     df=df,
#     L1=0.5,
#     L2=0.5,
#     L3=1,
#     excel_file_path=excel_file_path,
    
# )

# Run experiment
X, Y, metadata, f = run_experiment(
    n_replications=1,
    evalfn="DS3DrivenObjective",
    max_evals=150,
    n_initial_points=50,
    batch_size=1,
    # n_binary=96,
    n_categorical=0,
    n_continuous=0,
    init_with_k_spaced_binary_sobol=True,
    n_prototype_vectors=32,
    verbose=True,
    df=df,
    L1=1,
    L2=0,
    L3=2,
    # P=0,  
    excel_file_path=excel_file_path,
    
)

# %% save the data


# Convert tensors to numpy
X_np = X[0].cpu().numpy()
Y_np = Y[0].cpu().numpy()
selected_counts = X[0].sum(dim=1).cpu().numpy()

# mask = Y_np != 1e9

# Filter Y_np with this mask
# Y_filtered = Y_np[mask]

# Set your BO config values
max_evals_val = 150
n_initial_points_val = 50

# Build dataframe
data = {f'x_{i}': X_np[:, i] for i in range(X_np.shape[1])}
data['Y'] = Y_np
# data['Y_filtered'] = Y_filtered
data['#_selected_components'] = selected_counts
data['L1'] = [f.L1] * len(Y_np)
data['L2'] = [f.L2] * len(Y_np)
data['L3'] = [f.L3] * len(Y_np)
data['max_evals'] = [max_evals_val] * len(Y_np)
data['n_initial_points'] = [n_initial_points_val] * len(Y_np)
data['execution_time'] = f.execution_times
data['throughput'] = f.throughputs

# data['Penalty'] = [f.P] * len(Y_np)

df_results = pd.DataFrame(data)

# df_filtered = df_results[df_results['Y'] != 1e9]



# Save with incremented filename
base_name = "BO_results"
folder ="/local/data/falizadehziri_l/DS3/Results"
i = 1
while True:
    filename = f"{base_name}_{i}.xlsx"
    output_path = os.path.join(folder, filename)
    if not os.path.exists(output_path):
        break
    i += 1

df_results.to_excel(output_path, index=False)
print(f"Saved results to {output_path}")

# import numpy as np
# import pandas as pd
# import os

# def save_bo_results(X, Y, f, folder="/local/data/falizadehziri_l/DS3/Results", base_name="BO_results"):
#     """
#     Converts tensors to NumPy, filters out invalid Y values (1e9),
#     builds a DataFrame with BO results and saves it to an Excel file
#     with an incremented filename.
    
#     Parameters:
#         X (torch.Tensor): Input tensor (shape: [1, N, D])
#         Y (torch.Tensor): Output tensor (shape: [1, N])
#         f (object): An object containing .L1, .L2, .L3, .execution_times, .throughputs
#         folder (str): Folder to save the Excel file
#         base_name (str): Base name for output Excel file
    
#     Returns:
#         str: Path to the saved Excel file
#     """
#     # Convert tensors to NumPy
#     X_np = X[0].cpu().numpy()
#     Y_np = Y[0].cpu().numpy()
#     selected_counts = X[0].sum(dim=1).cpu().numpy()

#     # Filter out invalid entries
#     mask = Y_np != 1e9
#     Y_filtered = Y_np[mask]

#     # BO config values
#     max_evals_val = 150
#     n_initial_points_val = 50

#     # Build full results DataFrame
#     data = {f'x_{i}': X_np[:, i] for i in range(X_np.shape[1])}
#     data['Y'] = Y_np
#     data['#_selected_components'] = selected_counts
#     data['L1'] = [f.L1] * len(Y_np)
#     data['L2'] = [f.L2] * len(Y_np)
#     data['L3'] = [f.L3] * len(Y_np)
#     data['max_evals'] = [max_evals_val] * len(Y_np)
#     data['n_initial_points'] = [n_initial_points_val] * len(Y_np)
#     data['execution_time'] = f.execution_times
#     data['throughput'] = f.throughputs

#     df_results = pd.DataFrame(data)

#     # Also create filtered version (not saved here, but you can modify if needed)
#     df_filtered = df_results[df_results['Y'] != 1e9]

#     # Save with incremented filename
#     os.makedirs(folder, exist_ok=True)
#     i = 1
#     while True:
#         filename = f"{base_name}_{i}.xlsx"
#         output_path = os.path.join(folder, filename)
#         if not os.path.exists(output_path):
#             break
#         i += 1

#     df_results.to_excel(output_path, index=False)
#     print(f"Saved results to {output_path}")
#     return output_path


# %% Plot-1 the progression of the best objective value over iterations


max_evals = len(Y[0])
best_idx = torch.argmin(Y[0])
best_x = X[0][best_idx]  #This gives the best vector
best_objective_value = Y[0][best_idx]


a=[];
for i in range(max_evals):
  if Y[0][i]==1e9 :
    a.append(i)
   


plt.figure(figsize=(10, 6))
plt.plot(Y[0], marker='o', label="Objective Values")  # All objective values

# Highlight the best objective value with a red dot
plt.plot(best_idx, best_objective_value.item(), marker='o', color='red', markersize=8, label="Best Solution")

plt.xlabel("Iteration")
plt.ylabel("Objective Value (Total Space)")
plt.title("Bayesian Optimization Progress")
plt.grid(True)
plt.legend()  # Show the legend to differentiate the red dot
plt.show()

# %% Plot-2
import matplotlib.pyplot as plt


best_idx = torch.argmin(Y[0])
best_x = X[0][best_idx]  #This gives the best vector
best_objective_value = Y[0][best_idx]


a=[];
for i in range(max_evals):
  if Y[0][i]==1e9 :
    a.append(i)
    # print(i)

# Assuming train_obj is a numpy array
# Indices you want to exclude
exclude_indices = a

# Create a boolean mask to exclude the indices
mask = np.ones(len(Y[0]), dtype=bool)
mask[exclude_indices] = False

# Apply the mask to both indices and objective values
selected_indices = np.arange(len(Y[0]))[mask]
selected_train_obj = Y[0][mask]


# Plot the selected points
plt.figure(figsize=(10, 6))
plt.plot(selected_indices, selected_train_obj, marker='o', linestyle='-', label="Selected Points")
plt.plot(best_idx, best_objective_value.item(), marker='o', color='red', markersize=8, label="Best Solution")
plt.xlabel("Iteration")
plt.ylabel("Objective Value (Total Space)")
plt.title("Bayesian Optimization Progress (Excluding Specific Points)")
plt.grid(True)
plt.legend()
plt.show()

# # %% Plot-3: Number of Selected Components per Iteration

# selected_counts = []

# for i in range(len(Y[0])):
#     ## Uncomment for only feasible points
#     # if Y[0][i] == 1e9:
#     #     selected_counts.append(None)  # Mark infeasible solutions with None
#     #     continue
#     x_bool = X[0][i].round().to(torch.bool)
#     num_selected = x_bool.sum().item()
#     selected_counts.append(num_selected)

# # Create x-axis: iteration numbers excluding None
# iterations = [i for i, count in enumerate(selected_counts) if count is not None]
# selected_counts_filtered = [count for count in selected_counts if count is not None]

# plt.figure(figsize=(10, 6))
# plt.plot(iterations, selected_counts_filtered, marker='o', linestyle='-', label="# Selected Components")
# plt.xlabel("Iteration")
# plt.ylabel("Number of Selected Components (feasible points)")
# plt.title("Selected Components Over Iterations (Feasible points)")
# plt.grid(True)
# plt.legend()
# plt.show()
# # %% Plot-4
# ## not considering the penalties 

# import matplotlib.pyplot as plt
# import torch



# # Plot 1: Energy vs Space of Selected Sets
# energy_vals = []
# space_vals = []

# for x in X[0]:
#     x = x.to(torch.bool)
#     energy_vals.append(f.component_energy[x].sum().item())
#     space_vals.append(f.component_spaces[x].sum().item())

# # plt.figure(figsize=(6, 6))
# plt.scatter(energy_vals, space_vals, c='blue', label='Selected Points')
# plt.xlabel("Total Energy")
# plt.ylabel("Total Space")
# plt.title("Energy vs Space of Selected Component Sets")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.show()

# # Plot 2: Weighted Energy vs Weighted Space (L1 * space vs L2 * energy)
# weighted_energy = [f.L2 * e for e in energy_vals]
# weighted_space = [f.L1 * s for s in space_vals]

# # plt.figure(figsize=(6, 6))
# plt.scatter(weighted_energy, weighted_space, c='green', label='L1*Space vs L2*Energy')
# plt.xlabel(f"L2 * Energy (L2 = {f.L2})")
# plt.ylabel(f"L1 * Space (L1 = {f.L1})")
# plt.title("Weighted Energy vs Weighted Space")
# plt.grid(True)
# plt.tight_layout()
# plt.legend()
# plt.show()
# # %%

# ## not considering the penalties 

# from botorch.utils.multi_objective.pareto import is_non_dominated
# import numpy as np

# # Step 1: Stack the objectives: (space, energy) for minimization
# objectives = torch.tensor(list(zip(space_vals, energy_vals)))

# # Step 2: Invert for minimization (multiply by -1)
# objectives_min = -objectives

# # Step 3: Compute Pareto front (for minimization)
# is_pareto = is_non_dominated(objectives_min)

# # Step 4: Extract Pareto front points
# pareto_space = objectives[is_pareto][:, 0].numpy()
# pareto_energy = objectives[is_pareto][:, 1].numpy()

# # Step 5: Plot
# plt.scatter(energy_vals, space_vals, c='lightgray', label="All Points")
# plt.scatter(pareto_energy, pareto_space, c='red', label="Pareto Front")
# plt.xlabel("Total Energy")
# plt.ylabel("Total Space")
# plt.title("Pareto Front: Space vs Energy (Minimization)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# %%


import matplotlib.pyplot as plt
import numpy as np

# Extract the selected component count per iteration
num_selected_components = X[0].sum(dim=1).numpy()


plt.figure(figsize=(10, 5))
plt.plot(num_selected_components,'bo-')


plt.xlabel("Iteration")
plt.ylabel("Number of Selected Components")
plt.title("Feasible vs Infeasible Selections per Iteration")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# %%
