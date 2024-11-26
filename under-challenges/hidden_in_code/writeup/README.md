# hidden_in_code

### Disassembling the Bitcode

First, we disassemble the bitcode file to get the human-readable LLVM Intermediate Representation (IR):

```
$ llvm-dis targettarget.bc -o target.ll
```

Now, we can examine `target.ll` for any clues.

### Searching for Metadata

Scanning through `target.ll`, we notice metadata attached to instructions or basic blocks. Metadata in LLVM IR is denoted using exclamation marks (`!`) and can store arbitrary information.

For example:

```
; <label>:1:
  %0 = ...
  call void @some_function() !secret !0

!0 = !{!"a"}
```

### Formulating a Hypothesis

Based on our observations, we hypothesize that the flag is hidden within the metadata attached to basic blocks or instructions, using the key `!secret`.

## Developing a Solution

To extract the flag, we'll write a custom LLVM pass that traverses the IR, collects the metadata, and reconstructs the flag.

### Understanding the Challenge's Mechanism

Before writing our pass, it's helpful to understand how the metadata is embedded:

- **Flag Insertion**: The program uses a custom LLVM pass to insert the flag into the metadata of basic blocks, starting from the `main` function and traversing called functions.
- **Metadata Key**: The metadata is stored with the key `"secret"`.
- **Traversal Method**: The pass uses a breadth-first search (BFS) to traverse the call graph from `main`.

### Writing the Metadata Extraction Pass

We'll create an LLVM FunctionPass that:

1. Starts from the `main` function.
2. Performs a BFS traversal of the functions called.
3. In each basic block, checks for metadata with the key `"secret"`.
4. Collects the metadata strings to reconstruct the flag.

### Compiling the Pass

To compile the pass, we'll need LLVM's headers and libraries

```
clang++ -shared -fPIC -o libStringRecoveryPass.so StringRecoveryPass.cpp `llvm-config --cxxflags --ldflags --system-libs --libs core` -lLLVM
```

### Running the Pass

Use the `opt` tool to run our pass on the challenge bitcode:

```
opt -enable-new-pm=0 -load ./libStringRecoveryPass.so -string-recovery < target.bc > /dev/null
Recovered string: qwertyuikiy47b7OknG7pY7UIPLTHFn3aoEIyIFIopasdfghjklzxcvbnm0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ iCTF{M3t4d@7a_1s_wHat_Y0u_need} TTzGA86heRQB9BctimemhNO3piDbZdwhqsjQDMJNQJgSwiKYQrrl8yK8RrYToIQZTAmlVJxwugqRC52Ov81SyY8Q78jWO3538w1N2Uw9edB6G9lAO3ZKr4wqnKpNZoASlYrA1V1xe1kxFu4bp5zIBylQZx5WEAyz3QPLCBMwgya3HVy0elg5o0Pz4yXDjN4hWuXwQWo5w7f50o0BO5Fab2cj21cLXHhKmhKI4Qw9s1dBd30KvxksOdtjU5NaSwPaCBGsMznyUD3idZUKnqr4Y5fCjAhB7xMbbzbvfj6a9OIVrsyoorQLDvQBkZJHhRSSFfDXll1LrSOys0mLK9yqpaoJllcSwghty4mxTWFPBxR3Wtrdx3chh1ytzyRTcKKScHwhDUvFTmUVr36SZUDTqIqABaGvAEMiuqy7vxbqJyPuzcAlkejaJAi4Az2ZoPBVVn9WzqCjXqOmAcajOqpWY0lqRCWLuvwz9jQenyS6PMjhHaMs3kkfhYKLQ2q9r1do2xcVz0Y2JeS9gJ2RZfTc6PHDZUmKtgxriqbvQgShjkVSrwcdkzSWizdK
```

### Observing the Output

The pass should output the recovered flag:

```
iCTF{M3t4d@7a_1s_wHat_Y0u_need} 
```

