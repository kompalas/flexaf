import glob
import pandas as pd

# Define the path to the report files
report_files = glob.glob('./reports/reports_*_*.txt')
# report_files = glob.glob('../reports/reports_*.txt')

# Initialize an empty list to store the data
data = []

# Read each report file and extract the values
for report_file in report_files:
    with open(report_file, 'r') as file:
        line = file.readline().strip()
        parts = line.split('\t')
        print(parts)
        accuracy = parts[0].split(': ')[1]
        bitwidth = parts[1].split(': ')[1]
        window_size = parts[2].split(': ')[1]
        area = parts[3].split(': ')[1].split(' ')[0]
        power = parts[4].split(': ')[1].split(' ')[0]
        # print(parts[5])
        delay = parts[5].split(': ')[1].split(' ')[0]
        if len(parts) > 6:
            slack = parts[6].split("slack ")[1]
        else:
            slack = "?"
        data.append([accuracy, bitwidth, window_size, area, power, delay, slack])

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=['Accuracy', 'BITWIDTH', 'WINDOW_SIZE', 'Area (um^2)', 'Power (mW)', 'Delay (ns)', 'Slack'])
# df.sort_values(by='BITWIDTH', ascending=True, inplace=True)
df.sort_values(by=['WINDOW_SIZE', 'BITWIDTH'], ascending=True, inplace=True)

# Write the DataFrame to an Excel file
df.to_excel('reports/report_summary.xlsx', index=False)

print("Excel file 'report_summary.xlsx' has been created.")