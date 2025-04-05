import json
import os


def is_relevant_url(url: str) -> bool:
    """
    Returns True if the URL points to a page with meaningful text content,
    and False if it's a resource file like CSS, JS, images, etc.
    """
    url = url.lower()
    # Exclude known file types that don't contain content
    if any(
        url.endswith(ext)
        for ext in [
            ".css",
            ".js",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".ico",
            ".woff",
            ".ttf",
            ".eot",
            ".otf",
            ".map",
            ".jsf",
        ]
    ):
        return False

    # Exclude paths that clearly indicate layout/resource files (specific to CMS systems like DNN)
    if any(
        part in url
        for part in [
            "/skins/",
            "/containers/",
            "/layouts/",
            "/resources/",
            "/resource/",
            "/images/",
            "/fonts/",
            "/vendor/",
        ]
    ):
        return False

    # Otherwise, keep the URL
    return True


def filter_json_file(filepath, output_dir):
    """Filter the JSON file to keep only relevant URLs. text_by_page_url field of the json file is changed to
    contain only relevant URLs, and not resources such as css, javascript, image files.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_pages = data.get("text_by_page_url", {})
    filtered_pages = {
        url: text for url, text in original_pages.items() if is_relevant_url(url)
    }

    data["text_by_page_url"] = filtered_pages

    filename = os.path.basename(filepath)
    out_path = os.path.join(output_dir, filename)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(
        f"Filtered: {filename} (kept {len(filtered_pages)}/{len(original_pages)} pages)"
    )
