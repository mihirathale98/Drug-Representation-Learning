import json
import os

import requests
import sseclient

KEY = open('together_key.txt', 'r').read().strip()

def get_llm_response(input_prompt):
    url = "https://api.together.xyz/inference"
    model = "meta-llama/Llama-2-7b-chat-hf"
    input_prompt = input_prompt + '\n\n'

    payload = {
        "model": model,
        "prompt": input_prompt,
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stream_tokens": True,
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {KEY}",
    }

    response = requests.post(url, json=payload, headers=headers, stream=True)
    response.raise_for_status()

    client = sseclient.SSEClient(response)

    return client

# client = get_llm_response("hello")

# for event in client.events():
#     if event.data == "[DONE]":
#         break
#
#     partial_result = json.loads(event.data)
#     token = partial_result["choices"][0]["text"]
#     print(token, end="", flush=True)
