import openai
import os
from dotenv import load_dotenv
from thefuzz import fuzz
from tqdm import tqdm


def extract_keywords(query, model="gpt-4o-mini"):
    """
    Extract relevant keywords from a query using GPT-4o.
    """
    load_dotenv()
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"""
    Extract 3-5 specific keywords or key phrases from the following query. 
    Focus on entities, technical terms, and specific concepts that would be useful for document retrieval.
    Return only the keywords separated by commas, with no additional text.
    
    Query: {query}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0.3,
    )

    keywords_text = response.choices[0].message.content.strip()
    keywords = [k.strip() for k in keywords_text.split(",")]
    return keywords


def fuzzy_search(query_keywords, documents, threshold=70):
    """
    Perform fuzzy search on documents using extracted keywords.
    """
    results = []

    for doc in tqdm(documents, desc="Fuzzy searching"):
        content = doc.page_content.lower()
        max_score = 0

        for keyword in query_keywords:
            keyword = keyword.lower()

            if keyword in content:
                score = 100
            else:
                score = fuzz.partial_ratio(keyword, content)

            max_score = max(max_score, score)

        if max_score >= threshold:
            results.append((doc, max_score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in results]


def hybrid_retrieve_documents(query, vector_db=None, k=3, use_hybrid=True):
    """
    Retrieve documents using both vector similarity and keyword-based fuzzy search.
    """
    if vector_db is None:
        return []

    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    vector_docs = retriever.invoke(query)

    if not use_hybrid:
        return vector_docs

    all_docs = vector_db.get()["documents"]
    keywords = extract_keywords(query)
    fuzzy_docs = fuzzy_search(keywords, all_docs, threshold=70)[:k]

    # combine results
    seen_contents = set()
    combined_docs = []

    # add vector docs first (higher priority)
    for doc in vector_docs:
        if doc.page_content not in seen_contents:
            combined_docs.append(doc)
            seen_contents.add(doc.page_content)

    # add fuzzy docs
    for doc in fuzzy_docs:
        if doc.page_content not in seen_contents and len(combined_docs) < k * 2:
            combined_docs.append(doc)
            seen_contents.add(doc.page_content)

    # return top k combined results
    return combined_docs[:k]
