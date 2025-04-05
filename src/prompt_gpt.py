import openai
import os
from dotenv import load_dotenv


def get_gpt_response(prompt: str) -> str:
    """
    Get a response from OpenAI's GPT-4o model.

    Args:
        prompt: The text prompt to send to the model
        api_key: OpenAI API key. If None, will try to use OPENAI_API_KEY from environment

    Returns:
        The model's response as a string
    """

    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=30,
    )

    # Extract and return the response text
    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    load_dotenv()
    print("Only execute if sure! This already works.")
    assert False
    response = get_gpt_response("Tell me a short joke about programming.")
    print(response)
