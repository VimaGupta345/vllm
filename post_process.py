import pandas as pd

# Replace 'your_file.csv' with the path to your CSV file
input_file = 'unique_router_logits.csv'
# output_file = 'unique_router_logits.csv'

# Read the CSV file
df = pd.read_csv(input_file)

number_of_rows = df.shape[0]
print("Number of rows:", number_of_rows)

# # Drop duplicate rows
# unique_df = df.drop_duplicates()

# # Save the unique rows to a new CSV file
# unique_df.to_csv(output_file, index=False)

# print("Duplicate rows removed and unique rows saved to 'unique_rows.csv'")


# import pandas as pd
# import csv
# # Function to detect the delimiter in a CSV file
# def detect_delimiter(csv_file):
#     with open(csv_file, 'r', newline='', encoding='utf-8') as file:
#         sniffer = csv.Sniffer()
#         dialect = sniffer.sniff(file.readline())
#         return dialect.delimiter


# # Load the CSV file
# input_file = 'unique_router_logits.csv'  # Replace with your file path
# output_file = 'highlighted_file.csv'  # Output file
# delimiter = detect_delimiter(input_file)
# df = pd.read_csv(input_file)

# # Function to highlight top 2 values in Logit_x columns
# def highlight_top2(row):
#     logit_cols = [col for col in df.columns if col.startswith('Logit_')]
#     top2 = row[logit_cols].nlargest(2).index  # Get the indices of the top 2 values
#     new_row = row.copy()
#     for col in logit_cols:
#         if col in top2:
#             new_row[col] = f'**{row[col]}**'
#     return new_row

# # Apply the function to each row
# highlighted_df = df.apply(highlight_top2, axis=1)

# # Save the modified dataframe using the same delimiter
# highlighted_df.to_csv(output_file, index=False, sep=delimiter)

# print(f"File saved with top 2 values highlighted in '{output_file}'")