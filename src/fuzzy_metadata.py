from rapidfuzz import fuzz
import json
import os
from langchain.schema import Document
import tqdm


def fuzzy_find_in_text(text, keyword_list, threshold=75):
    matches = []
    for kw in keyword_list:
        score = fuzz.partial_ratio(kw.lower(), text.lower())
        if score >= threshold:
            matches.append(kw)
    return matches or ["unknown"]

def load_documents(json_file):
    """Loads the JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
      try:
          data = json.load(f)
          return data
      except json.JSONDecodeError:
          print(f"Error reading {json_file}, it may not be a valid JSON file.")
    return []



def fuzzy_is_meta(use_all_doc= True):
    
    TECHNOLOGY_KEYWORDS = [
        "robotics", "automation", "machine learning", "AI", "blockchain",
        "3D printing", "cloud computing", "big data", "additive manufacturing",
        "cybersecurity", "computer vision", "embedded systems", "augmented reality", "data analytics", "deep learning"
    ]

    SERVICES_KEYWORDS = [
        "logistics", "warehousing", "consulting", "engineering", "maintenance",
        "R&D", "quality assurance", "installation", "training", "support services",
        "custom manufacturing", "system integration", "product design", "supply chain services",
        "distribution", "transportation", "inventory management", "contract manufacturing",
        "technical support", "reverse logistics"
    ]

    MATERIALS_KEYWORDS = [
        "steel", "aluminum", "plastic", "rubber", "glass",
        "ceramics", "wood", "copper", "carbon fiber", "composites",
        "silicon", "polymer", "titanium", "magnesium", "resin",
        "PVC", "textile", "nylon", "stainless steel", "thermoplastics"
    ]

    PRODUCTS_KEYWORDS = [
        "auto components", "medical devices", "semiconductors", "batteries", "valves",
        "sensors", "electric motors", "pumps", "tools", "circuit boards",
        "robot arms", "controllers", "lighting systems", "furniture", "packaging",
        "industrial equipment", "hydraulic systems", "wires", "connectors", "pipes"
    ]

    INDUSTRIES_KEYWORDS = [
        "automotive", "aerospace", "electronics", "medical", "energy",
        "construction", "chemicals", "plastics", "pharmaceuticals", "textiles",
        "logistics", "metallurgy", "machinery", "furniture", "renewables",
        "mining", "agriculture", "defense", "automation", "packaging"
    ]

    REGIONS_KEYWORDS = [
        "Germany", "France", "Italy", "Switzerland", "Austria", "Netherlands", "Belgium",
        "Luxembourg", "Poland", "Czech Republic", "Spain", "Portugal", "United Kingdom",
        "Sweden", "Norway", "Denmark", "Finland", "Southern Europe", "DACH", "Benelux",
        "Nordics", "Eastern Europe", "Europe", "USA", "Canada", "China", "India", "Southeast Asia",
        "Middle East", "Latin America", "North Africa"
    ]
    cwd = os.getcwd()
    cwd = os.path.join(cwd, "data", "clean")
    files_in_folder = os.listdir(cwd)
    
    if use_all_doc==True:
        k= len(files_in_folder)
    else:
        k= 5

    documents = []
    for filename in tqdm.tqdm(files_in_folder[:k]):
        if filename.endswith('.json'):
            file_path = os.path.join(cwd, filename)
            doc = load_documents(file_path)

            if not doc or 'text_by_page_url' not in doc:
                continue

            for url, cleaned_text in doc['text_by_page_url'].items():
                metadata = {
                    "source": url,
                    "technologies": fuzzy_find_in_text(cleaned_text, TECHNOLOGY_KEYWORDS),
                    "services": fuzzy_find_in_text(cleaned_text, SERVICES_KEYWORDS),
                    "materials": fuzzy_find_in_text(cleaned_text, MATERIALS_KEYWORDS),
                    "products": fuzzy_find_in_text(cleaned_text, PRODUCTS_KEYWORDS),
                    "industries": fuzzy_find_in_text(cleaned_text, INDUSTRIES_KEYWORDS),
                    "regions": fuzzy_find_in_text(cleaned_text, REGIONS_KEYWORDS)
                }

                lan_doc = Document(page_content=cleaned_text, metadata=metadata)
                documents.append(lan_doc)
    return documents