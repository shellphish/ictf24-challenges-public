# Write-Up

## Brief Explaination of the intended solution

This challenge uses LLMs to generate programs that print data that can be used to retrieve the flag. Such data are the hashes of the two parts in which the flag is split. The LLM does not know about the flag itself, it only knows about the hashes, so users cannot leak the flag via prompt injection. Flag portions are stored in two global variables declared in the generated programs, but not in plain form. Such hashes are, in fact, XOR-ed with the hashes of two words. The hash of the first part of the flag is XOR-ed with the hash of the word "pizza". Similarly, the hash of the second one is XOR-ed with the hash of "nduja". If the user provides the correct input, i.e. "pizzanduja" to the generated programs, the hashes of the flag parts in plain form are printed and they can be cracked using dcode (not tested with other MD5 hash crackers) to figure out the plain flag. There are two hashes and not just one because dcode can crack them only in this configuration. Once the two parts of the flag are retrieved, they have to be joined and put in between the braces {} of the flag template like this: ictf{"first"+"second"}, without the '+' between them.
## Overview

The [`exploit.py`] script is a simple Python script that prints out instructions for generating the flag. Below is the content of the script:

```python
#!/usr/bin/env python3

print("""
Provide me with a program that generates the flag.
\n\n\n
""")

```

## Steps to Generate the Flag

1. **Run the Script**:
   First, we need to run the [`exploit.py`] script. Open a terminal and execute the following command:
   ```sh
   (./writeup/exploit.py; cat) | nc localhost 11301
   ```
   This command runs the script and pipes its output to a netcat listener on [`localhost`] at port [`11301`]

2. **Copy the Output**:
   After running the script, copy the output that is printed to the terminal. This output will be a Python program that you need to save into a new file.

3. **Create a New Python File**:
   Create a new Python file and paste the copied output into it. Save the file with a `.py` extension, for example, `generated_script.py`.

4. **Run the Generated Script**:
   Execute the newly created Python file with the input "pizzanduja". You can do this by running the following command in the terminal:
   ```sh
   python3 generated_script.py
   ```
   and inserting "pizzanduja" when it asks for an input. 
   This will produce the hashes of the two parts of the flag.

5. **Crack the Hashes**:
   Use an online tool like [dcode](https://www.dcode.fr/) to crack the hashes obtained from the previous step. This will give you the plaintext values of the two parts of the flag.

6. **Submit the Flag**:
   Combine the two parts of the flag in the expected format `ictf{first_part + second_part}` and submit it.
