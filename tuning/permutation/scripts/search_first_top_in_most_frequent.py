import sys

def sort_lines_by_second_column(file_path):
    """Sort lines in a file by the second column."""
    try:
        with open(file_path, 'r') as file:            lines = file.readlines()

        # Sort lines by the second column
        sorted_lines = sorted(lines, key=lambda line: float(line.split(',')[1].strip()), reverse=True)
        return sorted_lines

    except IndexError:
        print("Error: Some lines in the file do not have two columns.")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def search_first_column(sorted_lines, second_file_path):
    """Search for the first column of sorted lines in the second file."""
    try:
        with open(second_file_path, 'r') as file:
            second_file_lines = file.readlines()

        # Iterate through the sorted lines
        for line in sorted_lines:
            first_column = line.split(',')[0].strip()  # Get the first column

            # Search for the first column in the second file
            for line2 in second_file_lines:
                if first_column in line2:
                    print(f"{first_column}")
                    return  # Stop once a match is found

        # If no match is found
        print("No match found.")

    except FileNotFoundError:
        print(f"Error: The file '{second_file_path}' does not exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <file1> <file2>")
        sys.exit(1)

    # Get file names from command-line arguments
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    print(file1, end=":")

    # Step 1: Sort the lines in the first file by the second column
    sorted_lines = sort_lines_by_second_column(file1)

    # Step 2: Search for the first column of sorted lines in the second file
    search_first_column(sorted_lines, file2)

