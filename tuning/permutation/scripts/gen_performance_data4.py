import os
import subprocess
import itertools


# Constants
MIN_RANGE = 16
MAX_RANGE = 2**30
MAX_PRODUCT = 2**31
COMMAND = "./permutation_tuning_4"  # Replace with the actual command

dataTypes = ['F32', 'F16']

# Generate all permutations of [0, 1, 2, 3]
perms = list(itertools.permutations(["0", "1", "2", "3"]))

for perm in perms:
    outputDimsSpace = ' '.join(perm)
    outputDimsUnderscore = '_'.join(perm)
    # Loop through values of i1 and i2
    i1 = MIN_RANGE
    while i1 <= MAX_RANGE:
        i2 = MIN_RANGE
        while i2 <= MAX_RANGE:
            i3 = MIN_RANGE
            while i3 <= MAX_RANGE:
                i4 = MIN_RANGE
                while i4 <= MAX_RANGE:
                    if i1 * i2 * i3 * i4 >= MAX_PRODUCT:
                        # Skip this iteration if product exceeds 2^31
                        break

                    for dataType in dataTypes:
                        # Construct the output file name
                        output_file = f"data4/{dataType}_{i1}_{i2}_{i3}_{i4}_{outputDimsUnderscore}.txt"

                        # Construct the command
                        cmd = f"{COMMAND} {dataType} {i1} {i2} {i3} {i4} {outputDimsSpace}"

                        # Execute the command and save output to file
                        print(f"Running: {cmd}")
                        try:
                            with open(output_file, "w") as file:
                                subprocess.run(cmd, shell=True, check=True, stdout=file, stderr=subprocess.PIPE)
                        except subprocess.CalledProcessError as e:
                            print(f"Error while running command: {cmd}\n{e.stderr.decode()}")

                    # Increment i4
                    i4 *= 8

                # Increment i3
                i3 *= 8

            # Increment i2
            i2 *= 8

        # Increment i1
        i1 *= 8

