from func_file import function_bodies,function_explanations,function_descriptions
import chromadb
from chromadb.config import Settings

def Function(body, explanation, description):
    return {
        "body": body,
        "explanation": explanation,
        "description": description
    }
def debug(client,collection):
    print("What would you like to do ?")
    print("1. Dump the chromadb")
    print("2. Add a collection")
    print("3. Add a document")
    print("4. Go back")
    choice = int(input())
    if choice == 1:
        print(client.get())
    elif choice == 2:
        print("Please give me the name of the collection")
        name = input()
        client.create_collection(name)
    elif choice == 3:
        print("Enter the body of the function")
        body = input()
        try:
            logs = eval(body)
        except Exception as e:
            print("The function is invalid")
            print(e)
            return
        print("Enter the explanation of the function")
        explanation = input()
        ids = f"func_{len(collection.get()['ids']) + 1}"
        print("Using id as ", ids)
        collection.add(documents=[body], metadatas=[{"func_paragraph" :explanation,"func_name":ids}], ids=[ids])
        print("Function added successfully and execution logs are as follows:")
        print(logs)
    elif choice == 4:
        return

def query(collection):
    print("Please enter the function you would like to query")
    query = input()
    results = collection.query(query_texts=[query],n_results=2)
    if len(results) == 0:
        print("No results found")
        return
    print(f"We were able to find {len(results['ids'][0])} results")
    for i in range(len(results['ids'][0])):
        print(f"Result {i+1}:")
        explanation = results['metadatas'][0][i]
        print(f"Name of the function: {explanation['func_name']}")
        print(f"Explanation: {explanation['func_paragraph']}")
    print("Would you like to see the body of a function ?")
    print("1. Yes")
    print("2. No")
    choice = int(input())
    if choice == 1:
        for doc in results['documents'][0]:
            print(doc)
    else:
        return

def main():
    assert(len(function_bodies) == len(function_explanations) == len(function_descriptions))
    functions = []
    for i in range(len(function_bodies)):
        functions.append(Function(function_bodies[i], function_explanations[i] ,""))
    client = chromadb.Client()
    collection = client.create_collection(name="python_functions")
    for func in functions:
        collection.add(
        documents = [func["body"]],
        metadatas = [func["explanation"]],
        ids = [f"func_{functions.index(func)}"]
    )
    print("Welcome to the Python Guide book! Powered by ChromaDB")
    print("This program has a list of basic python functions and their explanations.")
    print("You can access the functions by providing what you want to do")
    print("Example usage is as follows:")
    print("""
          ----------------------------------------------
Welcome to the Python Guide book! Powered by ChromaDB
This program has a list of basic python functions and their explanations.
You can access the functions by providing what you want to do
Example usage is as follows:
Now what would you like to do ?
1. Query for a function
2. Exit
1
Please enter the function you would like to query
add
We were able to find 2 results
Result 1:
Name of the function: add
Explanation: The `add(a, b)` function takes two numbers (`a` and `b`) and adds them together. It’s like combining two values to get their total. If you call `add(3, 5)`, it will give you `8`.
Result 2:
Name of the function: sum_of_cubes
Explanation: The `sum_of_cubes(n)` function calculates the sum of cubes of the first `n` natural numbers. For example, `sum_of_cubes(3)` will return `1^3 + 2^3 + 3^3 = 36`.
Would you like to see the body of a function ?
1. Yes
2. No
1

def add(a, b):
    return a + b


def sum_of_cubes(n):
    return sum(i**3 for i in range(1, n+1))          
            ----------------------------------------------
""")



    print("Now what would you like to do ?")
    print("1. Query for a function")
    print("2. Exit")
    choice = int(input())
    if choice == 1:
        query(collection)
    elif choice == 2:
        print("Goodbye!")
        exit(0)
    elif choice == 1337:
        debug(client,collection)
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()