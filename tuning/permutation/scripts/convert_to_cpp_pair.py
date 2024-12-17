import re

def convert_string(input_string):
    """Converts a string in the format 'F32_16_16_0_1.txt:2_256_64_64_4_4_0_1_4_4'
       to a string representation of a dictionary-like structure.
    """
    try:
        filename, numbers_str = input_string.split(":")
        numbers = [int(x) for x in numbers_str.split("_")]

        # Extract parts from filename
        parts = filename[:-4].split("_")  # Remove ".txt"
        data_type = '32F' if parts[0] == 'F32' else '16F'
        rest_of_filename = "_".join(parts[1:])

        output_string = f'{{"HIP_R_{data_type}_{rest_of_filename}", {{{numbers[1]}, {numbers[2]}, {numbers[3]}, {numbers[4]}, {numbers[5]}, {{{numbers[6]}, {numbers[7]}}}, {numbers[8]}, {numbers[9]}}}}},'
        return output_string

    except (ValueError, IndexError):
        return None  # Return None for invalid lines


def convert_file(input_filename, output_filename):
    """Converts lines from an input file and writes the converted lines to an output file."""
    try:
        with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
            for line in infile:
                line = line.strip()  # Remove leading/trailing whitespace
                if line: #skip empty lines
                    converted_line = convert_string(line)
                    if converted_line: #skip invalid lines
                        outfile.write(converted_line + '\n')
                    else:
                        print(f"Warning: Invalid format in line: {line}")
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
input_filename = "best_instances.txt" #set the input file name
output_filename = "cpp_lut_items.txt"
convert_file(input_filename, output_filename)

print(f"Conversion complete. Check '{output_filename}'")

#print output file content
#  with open(output_filename, "r") as outfile:
    #  print(f"{output_filename} content:")
    #  for line in outfile:
        #  print(line, end="") #prevent double newlines
