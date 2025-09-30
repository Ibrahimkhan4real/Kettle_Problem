import torch
import sys

# Add this line to force PyTorch to print the full tensor
torch.set_printoptions(threshold=float('inf'))

# --- Script Starts Here ---

# Define the path to your file
file_path = 'optimizer_state_corrupt.pth' # <--- CHANGE THIS TO YOUR FILENAME

try:
    # Load the state_dict from the file
    loaded_state_dict = torch.load(file_path, map_location=torch.device('cpu'))
    print(f"--- Successfully loaded {file_path} ---\n")

    # Check if the loaded object is a dictionary (like a state_dict)
    if not isinstance(loaded_state_dict, dict):
        print(f"Error: The file {file_path} does not contain a state_dict (dictionary).")
        print(f"It contains an object of type: {type(loaded_state_dict)}")
        sys.exit()

    # Iterate through all key-value pairs in the dictionary
    # This is the robust part - it makes no assumptions about key names
    print("--- Inspecting all tensors in the file ---")
    for key, tensor_value in loaded_state_dict.items():
        print(f"\n--- Layer: {key} ---")
        #print(f"Shape: {tensor_value.shape}")
        print(f"Values:\n{tensor_value}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")