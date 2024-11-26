from gpt import Assistant


def print_banner():
    print("*********************")
    print("*  Welcome to iCTF  *")
    print("*********************")
    print("\nDo you have any questions about iCTF? Try out our new virtual assistant!")
    print("Ask me anything, and I will try to help you out.\n")
    print("For example, you can ask: `When does iCTF 2024 end?`.\n")


def main():
    print_banner()

    question = input("Enter your question: ")

    if len(question) > 30:
        print("Your question is too long!")
        return

    print("Let me find the answer for you...\n\n", flush=True)

    answer = Assistant.ask(question)

    if answer:
        print(answer)
    else:
        print("I'm sorry, I don't know the answer to that question.")

    print("See you next time!", flush=True)


if __name__ == "__main__":
    main()
