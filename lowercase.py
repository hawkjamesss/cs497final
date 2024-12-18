import csv

def make_lowercase_copy(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            for row in reader:
                # Convert all elements of the row to lowercase
                lower_row = [cell.lower() for cell in row]
                writer.writerow(lower_row)

# Example usage
input_csv = 'output.csv'    # Replace with your input file name
output_csv = 'devall.csv'  # Replace with your desired output file name
make_lowercase_copy(input_csv, output_csv)