import os
import subprocess
import itertools


# Constants
MIN_RANGE = 16
MAX_RANGE = 2**30
MAX_PRODUCT = 2**31
COMMAND = "./permutation_tuning_3"  # Replace with the actual command

dataTypes = ['F32', 'F16']

# Generate all permutations of [0, 1, 2]
perms = list(itertools.permutations(["0", "1", "2"]))

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
                if i1 * i2 *i3 >= MAX_PRODUCT:
                    # Skip this iteration if product exceeds 2^31
                    i3 *= 2
                    break

                for dataType in dataTypes:
                    # Construct the output file name
                    output_file = f"data3/{dataType}_{i1}_{i2}_{i3}_{outputDimsUnderscore}.txt"

                    # Construct the command
                    cmd = f"{COMMAND} {dataType} {i1} {i2} {i3} {outputDimsSpace}"

                    # Execute the command and save output to file
                    print(f"Running: {cmd}")
                    try:
                        with open(output_file, "w") as file:
                            subprocess.run(cmd, shell=True, check=True, stdout=file, stderr=subprocess.PIPE)
                    except subprocess.CalledProcessError as e:
                        print(f"Error while running command: {cmd}\n{e.stderr.decode()}")

                # Increment i3
                i3 *= 2

            # Increment i2
            i2 *= 2

        # Increment i1
        i1 *= 2

