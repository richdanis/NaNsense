import os
import json
import random
import openai
from dotenv import load_dotenv
from tqdm import tqdm


def load_random_json(directory="data/clean"):
    """
    Load a random JSON file from the specified directory.

    Args:
        directory: Directory containing JSON files

    Returns:
        Loaded JSON data
    """
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {directory}")

    random_file = random.choice(json_files)
    file_path = os.path.join(directory, random_file)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data, random_file


def get_random_page_content(json_data):
    """
    Extract content from a random page in the JSON data.

    Args:
        json_data: Loaded JSON data with text_by_page_url field

    Returns:
        Tuple of (page_url, page_content)
    """
    if not json_data.get("text_by_page_url"):
        raise ValueError("JSON data does not contain text_by_page_url field")

    urls = list(json_data["text_by_page_url"].keys())
    if not urls:
        raise ValueError("No pages found in text_by_page_url")

    random_url = random.choice(urls)
    page_content = json_data["text_by_page_url"][random_url]

    return random_url, page_content


def generate_qa_pair(page_content, url, website_url, model="o1"):
    """
    Generate a question-answer pair based on the page content using OpenAI.
    The question should be answerable directly from the provided content and only contain a single word.

    Args:
        page_content: Text content of the page
        url: URL of the page
        website_url: Base URL of the website
        model: OpenAI model to use

    Returns:
        Dictionary containing the generated question and answer
    """
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Truncate content if it's too long (to avoid token limits)
    max_content_length = 8000
    if len(page_content) > max_content_length:
        page_content = page_content[:max_content_length]

    prompt = f"""
You are an expert at creating high-quality question-answer pairs for testing retrieval-augmented generation (RAG) systems.

WEBSITE: {website_url}
PAGE URL: {url}
PAGE CONTENT:
{page_content}

Based on the content above, create ONE specific question and its corresponding answer. The question should:
1. Be answerable directly from the provided content
2. Be specific enough to test precise information retrieval
3. Require understanding of the content (not just keyword matching)
4. Be clear and unambiguous

The answer should:
1. Be comprehensive but concise
2. Include only information present in the content
3. Be factually accurate

Format your response as a JSON object with "question" and "answer" fields only:
{{
  "question": "Your question here",
  "answer": "Your answer here"
}}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    # Parse the response
    try:
        result = json.loads(response.choices[0].message.content)
        return result
    except json.JSONDecodeError:
        # If parsing fails, return a default error response
        return {
            "question": "ERROR: Failed to generate valid question",
            "answer": "ERROR: Failed to generate valid answer",
        }


def create_benchmark(num_samples=50, output_file="benchmark.json"):
    """
    Create a benchmark dataset with question-answer pairs.

    Args:
        num_samples: Number of QA pairs to generate
        output_file: Path to save the benchmark file

    Returns:
        None
    """
    # Load environment variables for API key
    load_dotenv()

    benchmark_data = []

    for _ in tqdm(range(num_samples), desc="Generating QA pairs"):
        try:
            # Load a random JSON file
            json_data, filename = load_random_json()
            website_url = json_data.get("url", "Unknown")

            # Get random page content
            page_url, page_content = get_random_page_content(json_data)

            # Generate QA pair
            qa_pair = generate_qa_pair(page_content, page_url, website_url)

            # Add metadata
            benchmark_item = {
                "website": website_url,
                "page_url": page_url,
                "source_file": filename,
                "question": qa_pair.get("question"),
                "answer": qa_pair.get("answer"),
            }

            benchmark_data.append(benchmark_item)

        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            continue

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(
        f"Benchmark created with {len(benchmark_data)} QA pairs and saved to {output_file}"
    )


if __name__ == "__main__":
    # Set the number of samples and output file
    num_samples = 50  # Adjust as needed
    output_file = "benchmark.json"

    # Create the benchmark
    create_benchmark(num_samples, output_file)
