import json
import os
import openai
from dotenv import load_dotenv


def load_prompts(file_path="src/prompts.json"):
    """
    Load prompts from a JSON file.

    Args:
        file_path: Path to the JSON file containing prompts

    Returns:
        Dictionary of prompts
    """
    with open(file_path, "r") as f:
        return json.load(f)


def generate_answer(query, retrieved_texts, prompt_template, model="gpt-4o"):
    """
    Get a response from OpenAI's models, default is gpt-4o.

    Args:
        prompt: The text prompt to send to the model
        api_key: OpenAI API key. If None, will try to use OPENAI_API_KEY from environment

    Returns:
        The model's response as a string
    """
    # Initialize the OpenAI client
    load_dotenv()
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Join retrieved texts into a single context string
    if isinstance(retrieved_texts, list):
        context = "\n\n".join(retrieved_texts)
    else:
        context = str(retrieved_texts)

    formatted_prompt = prompt_template.format(context=context, query=query)
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": formatted_prompt}],
        max_tokens=30,
    )

    # Extract and return the response text
    return response.choices[0].message.content
