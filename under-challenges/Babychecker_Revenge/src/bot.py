import os
from openai import OpenAI
import json

def checker(url):
    client = OpenAI(
        api_key = os.environ.get("OPENAI_API_KEY"),
        base_url = os.environ.get("OPENAI_BASE_URL")
    )
    check_prompt = f'''Imagine you are an expert SSRF checker that knows every thing about SSRF.
You should only focus on SSRF, but should not block common requests to external websites.
The link is : "{url}"

If the link is specious, then return result as True. 
If the link is common, then return result as False.

You should output url, result and reason in the following format.
If the result is True, then sanitize the url , if can't, just set the url as https://example.com
Response with JSON format.
{{
"url":{{}},
"result":{{}},
"reason":{{}}
}} '''
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": check_prompt,
            }
        ],
        model="oai-gpt-3.5-turbo",
        max_tokens=1000,
        response_format={"type": "json_object"}
    )
    output = response.choices[0].message.content
    data = json.loads(output)
    return data