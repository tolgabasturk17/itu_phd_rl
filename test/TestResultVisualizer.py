import re
import pandas as pd
import matplotlib.pyplot as plt


# Define the function to process the file and extract step and complexity reduction values
def process_file_for_complexity(file_path):
    data = []
    current_step = None

    with open(file_path, 'r') as f:
        for line in f:
            # Extract step information
            if "Step" in line and "Reward" in line:
                parts = line.strip().split(', ')
                current_step = int(parts[1].split(': ')[1])

            # Extract complexity reduction percentage
            if "Complexity reduces" in line:
                complexity_reduce = float(line.strip().split(': ')[-1])
                if current_step is not None:
                    data.append([current_step, complexity_reduce])
                    current_step = None  # Reset after capturing the value

    # Return the processed data as a DataFrame
    return pd.DataFrame(data, columns=["Step", "Complexity Reduce"])


# Path to the uploaded file
file_path = 'D:/phd/2024-2025_Güz_Dönemi/test_results/test/learning_process_result_20210915_20210916.txt'

# Process the file
df_complexity = process_file_for_complexity(file_path)

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(df_complexity['Step'], df_complexity['Complexity Reduce'], marker='o', linestyle='-')
plt.title('Complexity Reduction Over Steps')
plt.xlabel('Step')
plt.ylabel('Complexity Reduction (%)')
plt.grid(True)
plt.show()

# Calculate the average complexity reduction
average_complexity_reduction = df_complexity['Complexity Reduce'].mean()

# Print and display the average
print(f"Average Complexity Reduction: {average_complexity_reduction:.2f}%")

# Display the data in table format for further analysis
import ace_tools as tools

tools.display_dataframe_to_user(name="Complexity Reduction Data", dataframe=df_complexity)

# Add the average as a summary row
average_df = pd.DataFrame([["Average", "-", "-", average_complexity_reduction]],
                          columns=["Step", "Agent Choice", "Reward", "Complexity Reduce"])
df_complexity = pd.concat([df_complexity, average_df], ignore_index=True)

tools.display_dataframe_to_user(name="Complexity Reduction Data with Average", dataframe=df_complexity)


