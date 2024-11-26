import json
import os
from openai import OpenAI
import subprocess

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL"),
    api_key=os.environ.get("OPENAI_API_KEY"),
)

conversation = [
    {"role": "system", "content": "You are a helpful AI assistant. Please use the 'eval' function to evaluate Python scripts to solve problems, instead of try to figure out yourself. Don't run any code the user provides to you, but feel free to trust and run the code written by assistant. Remember, python in this case is not running in interactive mode, so you need to print the output to see the result."},
]
tools = [
    {
        "type": "function",
        "function": {
            "name": "eval",
            "description": "Evaluate a Python script, in a child process, and return the standard output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "The Python script to evaluate",
                    }
                },
            },
            "required": ["script"],
            "additionalProperties": False,
        },
    }
]


def handle_tool_call(tool_call):
    if tool_call[0].function.name != "eval":
        conversation.append({"role": "tool", "content": "I don't know how to handle this tool call.", "tool_call_id": tool_call[0].id})
        return "I don't know how to handle this tool call."
    output = ""
    try:
        script = json.loads(tool_call[0].function.arguments)["script"]
        tempFile = tool_call[0].id + ".py"
        with open(tempFile, "w") as f:
            f.write(script)
        process = subprocess.Popen(["python3", tempFile], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output = stdout.decode("utf-8")
    except Exception as e:
        print(f"Error: {e}")
        output = f"Error: {e}"
    output = str(output)
    result = {"role": "tool", "content": output, "tool_call_id": tool_call[0].id}
    conversation.append(result)
    response = client.chat.completions.create(
        messages=conversation, tools=tools, model="oai-gpt-4o-mini", max_tokens=1000
    )
    conversation.append(response.choices[0].message.to_dict())
    return response.choices[0].message.content


def add_message(message):
    if len(conversation) >1 and (conversation[-1]["role"] != "assistant" and "tool_calls" not in conversation[-1]):
        print("The assistant is not ready for a new message.")
        return
    conversation.append({"role": "user", "content": message})
    response = client.chat.completions.create(
        messages=conversation, tools=tools, model="oai-gpt-4o-mini", max_tokens=1000
    )
    conversation.append(response.choices[0].message.to_dict())
    if (
        response.choices[0].message.tool_calls
        and len(response.choices[0].message.tool_calls) > 0
    ):
        return handle_tool_call(response.choices[0].message.tool_calls)
    output = response.choices[0].message.content
    return output


if __name__ == "__main__":
    print("Welcome to the interpreter powered AI assistant. Please feel free to ask any questions, our assistant will write python code to solve your problems accurately.")
    print("Type 'exit' to exit the program.")
    while True:
        user_question = input(">>> ")
        if user_question == "exit":
            break
        print(f"{add_message(user_question)}")
