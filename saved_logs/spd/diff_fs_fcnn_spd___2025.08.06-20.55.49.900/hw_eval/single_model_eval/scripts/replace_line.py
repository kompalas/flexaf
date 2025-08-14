import sys

def replace_line_in_file(file_path, target_line, replacement_line):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Open the file again in write mode to overwrite it
    with open(file_path, 'w') as file:
        for line in lines:
            # If the current line matches the target line, replace it
            if target_line in line:
                file.write(replacement_line + '\n')
            else:
                file.write(line)

# Usage
file_path = './sim/top_tb.v'  # Specify the file path
target_line = 'parameter CLK_PERIOD = '  # Line to be replaced.
CLK = sys.argv[1]
replacement_line = f"\tparameter CLK_PERIOD = {CLK};"  # Replacement line

replace_line_in_file(file_path, target_line, replacement_line)
