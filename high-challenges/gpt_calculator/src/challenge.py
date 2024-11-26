#! /usr/bin/env python3

import os
import subprocess as sp
import tempfile
import openai
from rich.console import Console
from rich.rule import Rule

BASE_URL = os.getenv("OPENAI_BASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL")

MESSAGES = []
INITIAL_PROMPT = """
You are a helpful calculator assistant. You are asked to help users with their math problems. Please don't do other tasks. You should generate a python code in a code block (```) to solve user's query. Please only return the code. If the user's query is not a math problem, you should just write a code print "1 + 1 = 2".
"""

console = Console()


def input_multiline():
	contents = ""
	while contents[-3:] != "\n\n\n":
		contents += input() + "\n"
	return contents.strip("\n\n\n")


def extract_code(msg):
	if "```python" in msg:
		code = msg.split("```python")[1].split("```")[0].strip()
		return code
	if "```" in msg:
		code = msg.split("```")[1].split("```")[0].strip()
		return code
	
	return ""
	


def run_code(code):
	# run the python code with user "gpt" and return the output if not failed
	with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
		f.write(code)
		f.close()
		fname = f.name
		os.chmod(fname, 0o644)
		output = ""
		try:
			out = sp.check_output(["su", "gpt", "-c", f"python3 {fname}"], timeout=10, cwd="/gpt")
			output = out.decode("utf-8")
		except Exception as e:
			output = "Sorry I don't know, please try again."
		os.remove(fname)
	
	return output


def main():
	client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
	MESSAGES.append(dict(role='system', content=INITIAL_PROMPT))

	console.print(Rule("Your prompt: (\\n\\n\\n to submit)"))
	prompt = input_multiline()
	MESSAGES.append(dict(role='user', content=prompt))
	console.print(f"[red]Received Prompt: {prompt}")
	console.print("[red]Processing...")
	console.print(Rule())
	
	response = client.chat.completions.create(
		model=MODEL, 
		messages=MESSAGES,
		max_tokens=512,
	)

	response_message = response.choices[0].message.content
	code = extract_code(response_message)
	if code:
		res = run_code(code)
	else:
		res = "Sorry I don't know, please try again."
 
	console.print(f"[green]Answer: {res}")


if __name__ == "__main__":
	intro = """I am a calculator! What can I help you with today?"""
	console.print(Rule("Welcome to the GPT Calculator challenge!"))
	try:
		console.print(intro)
		main()
	except KeyboardInterrupt:
		pass
	except openai.RateLimitError:
		# IMPORTANT: handle rate limit error
		console.print("Sorry you have reached the rate limit. Please try again later.")

	console.print()
	console.print(Rule())
	console.print("Alright, bye!")
