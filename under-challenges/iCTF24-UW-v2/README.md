# CTF Challenge: Prompt Injection - InvisiblePDFlag

## Overview

Welcome to the **Prompt Injection Challenge**! In this challenge, you will be tasked with finding a hidden flag embedded inside a resume. The flag has been cleverly concealed using a **prompt injection attack**, where a piece of **invisible text** is added to the document. Your mission is to detect this invisible text and extract the hidden flag using **Large Language Models (LLMs)** or other analysis techniques.

## Challenge Description

You’ve been provided with a PDF resume (`resume.pdf`) which appears to be a normal document. However, a recent security audit revealed vulnerabilities to **prompt injection** in the system used to process this resume. 

The **flag** has been hidden within the resume using this technique, and it's your job to find it.

### Objective
- Extract the **hidden flag**.

### File Provided:
- `resume.pdf`: A standard-looking resume that contains the hidden flag.

## Instructions

1. **Analyze the Resume**: Open and examine the provided resume (`resume.pdf`). At first glance, it might look like an ordinary resume, but the flag is hidden inside.
   
2. **Think Like an Attacker**: Consider how someone could inject hidden text into a file to manipulate **automated systems** like LLMs or AI tools.

3. **Use an LLM**: You are encouraged to use a tool that leverages **Large Language Models (LLMs)** to help you read and analyze the contents of the resume. LLMs may reveal content that’s hidden to the naked eye but visible to text-analysis software.


### Tips

- **Not Everything is Visible**: Consider that there could be **hidden content** that a traditional PDF viewer might not show but an LLM or automated system could detect.
  
- **Automated Analysis**: LLMs might see content that isn’t intended for humans, such as text designed to manipulate software behavior.

## Tools

To successfully solve this challenge, you may want to use the following tools:
- **LLM-based PDF analyzers**: Tools like ChatGPT, OpenAI API, or other LLM-based applications that can process and summarize text from PDFs.
- **Manual Investigation**: Carefully read and inspect the resume, paying attention to any areas that might seem redundant or excessively descriptive.


## Hints
- Not everything is as it seems—some content is intended to be ignored by the human reader but not by software.
- Use an LLM to summarize or analyze the PDF and detect the **injected text**.

Good luck, and happy hunting!
