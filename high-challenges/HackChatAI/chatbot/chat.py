def get_answer(client, question):
    response = client.chat.completions.create(
        model="oai-gpt-4o-mini", # Change this to `gpt-4o-mini` if you are using openAI API , here we are proxying through LiteLLM for our deployment
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content
