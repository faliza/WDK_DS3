import numpy as np

# Load original file
file_path = "bo_input_vectors.npz"
with np.load(file_path) as data:
    x = data["x"].copy()
    type_features = data["type_features"]
    task_tags = data["task_tags"]
# %%
print("Original vector:", task_tags.shape,x.shape,type_features.shape)

# Modify first vector
indices_to_enable = [31]
x[:] = 0
x[indices_to_enable] = 1
# %%


# Save as a new file
save_path = "/local/data/falizadehziri_l/DS3/bo_input_vectors.npz"

np.savez(save_path, x=x, type_features=type_features, task_tags=task_tags)

# Reload and check
check = np.load(save_path)
print("Saved vector:", check["x"][31])
