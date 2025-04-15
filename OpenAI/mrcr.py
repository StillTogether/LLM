import os
from dotenv import load_dotenv

from huggingface_hub import hf_hub_download
import pandas as pd
from openai import OpenAI
import json
from difflib import SequenceMatcher
import tiktoken

load_dotenv()


MAX_CONTEXT_WINDOW= 1000000
MODEL= "gpt-4.1"

dataset = pd.read_parquet(
    hf_hub_download(repo_id="openai/mrcr", filename="2needle.parquet", repo_type="dataset")
)
client = OpenAI(api_key=os.environ["OPENAI_KEY"])
enc = tiktoken.get_encoding("o200k_base")

def grade(response, answer, random_string_to_prepend) -> float:
    """
    Compare response and answer.
    """
    if not response.startswith(random_string_to_prepend):
        return 0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())

def n_tokens(messages : list[dict]) -> int:
    """
    Count tokens in messages.
    """
    return sum([len(enc.encode(m["content"])) for m in messages])

for index, row in dataset.iterrows():
    messages = json.loads(row["prompt"])
    if n_tokens(messages) > MAX_CONTEXT_WINDOW:
        continue
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    response = completion.choices[0].message.content
    print(grade(response, row["answer"], row["random_string_to_prepend"]))
