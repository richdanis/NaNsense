import os
import json
import random
import openai
from dotenv import load_dotenv
from tqdm import tqdm


prompt_template = """
You are an expert at creating high-quality question-answer pairs for testing retrieval-augmented generation (RAG) systems.

WEBSITE: {website_url}
PAGE URL: {url}
PAGE CONTENT:
{page_content}

Based on the content above, create ONE specific question and its corresponding single-word answer. The question should:
1. Be answerable directly from the provided content
2. Be specific enough to test precise information retrieval
3. Require understanding of the content (not just keyword matching)
4. Be clear and unambiguous

The answer should:
1. Be comprehensive but concise
2. Include only information present in the content
3. Be factually accurate
4. Be a single word

Format your response as a JSON object with "question" and "answer" fields only:
{{
  "question": "Your question here",
  "answer": "Your answer here"
}}
"""


def load_random_json(directory="data/clean"):
    """
    Load a random JSON file from the specified directory.
    """
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]
    random_file = random.choice(json_files)
    file_path = os.path.join(directory, random_file)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data, random_file


def generate_qa_pair(page_content, url, website_url, model="o1"):
    """
    Generate a question-answer pair based on the page content using o1.
    The question should be answerable directly from the provided content and only contain a single word.
    """
    # Initialize the OpenAI client
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Truncate content if it's too long (to avoid token limits)
    max_content_length = 8000
    if len(page_content) > max_content_length:
        page_content = page_content[:max_content_length]

    prompt = prompt_template.format(
        website_url=website_url, url=url, page_content=page_content
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    # Parse the response
    result = json.loads(response.choices[0].message.content)
    return result


def create_benchmark(num_samples=50, output_file="benchmark.json"):
    """
    Create a benchmark dataset with question-answer pairs.
    """
    load_dotenv()

    benchmark_data = []

    for _ in tqdm(range(num_samples), desc="Generating QA pairs"):
        try:
            json_data, filename = load_random_json()
            website_url = json_data.get("url", "Unknown")

            urls = list(json_data["text_by_page_url"].keys())
            page_url = random.choice(urls)
            page_content = json_data["text_by_page_url"][page_url]

            # generate QA pair
            qa_pair = generate_qa_pair(page_content, page_url, website_url)

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

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(
        f"Benchmark created with {len(benchmark_data)} QA pairs and saved to {output_file}"
    )


if __name__ == "__main__":
    num_samples = 50
    output_file = "benchmark.json"

    # create the benchmark
    create_benchmark(num_samples, output_file)
