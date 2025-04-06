import json
import os

from .resources_words import (
    business_url_paths,
    banned_extenstions,
    banned_url_paths,
    stopwords,
    languages,
)


def remove_stopwords(text):
    """
    Remove stopwords and language-specific words from the text to reduce size.
    """
    words = text.split()
    filtered_words = [
        word for word in words if (word not in stopwords and word not in languages)
    ]
    return " ".join(filtered_words)


def cut_first_k(text, percent):
    """
    Discard the first 10% of the words, since in the beginning there is a lot of repetition.
    """
    if percent == 1:
        return text

    words = text.split()
    k = int(len(words) * percent)
    return " ".join(words[k:])


def is_relevant_url(url: str) -> bool:
    """
    Returns True if the URL points to a page with meaningful text content,
    and False if it's a resource file like CSS, JS, images, etc.
    """

    url = url.lower()

    # keep the base URL
    if url.endswith(".com/"):
        return True

    # remove style links
    if any(url.endswith(ext) for ext in banned_extenstions) or any(
        ext in url for ext in banned_extenstions
    ):
        return False

    # remove links that clearly indicate layout/resource files (specific to CMS systems like DNN)
    if any(part in url for part in banned_url_paths):
        return False

    # keep URLs that are likely to contain meaningful content
    # this is a heuristic and may need to be adjusted based on the specific dataset
    if any(part in url for part in business_url_paths):
        return True

    # otherwise, remove the URL
    return False


def filter_json_file(filepath, output_dir, debug=False):
    """
    Filter the JSON file to keep only relevant URLs. text_by_page_url field of the json file is changed to
    contain only relevant URLs, and not resources such as css, javascript, image files.
    """

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_pages = data.get("text_by_page_url", {})
    filtered_pages = {
        url: cut_first_k(remove_stopwords(text), percent=0.1)
        for url, text in original_pages.items()
        if is_relevant_url(url)
    }

    data["text_by_page_url"] = filtered_pages

    filename = os.path.basename(filepath)
    out_path = os.path.join(output_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if debug:
        print(
            f"Filtered: {filename} (kept {len(filtered_pages)}/{len(original_pages)} pages)"
        )
