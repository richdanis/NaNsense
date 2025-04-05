import re
import openai
import os
from dotenv import load_dotenv
from thefuzz import fuzz
from tqdm import tqdm


def extract_keywords(query, model="gpt-4o-mini"):
    """
    Extract relevant keywords from a query using GPT-4o.

    Args:
        query: User query string
        model: OpenAI model to use

    Returns:
        List of keywords
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

    # Extract and clean keywords
    keywords_text = response.choices[0].message.content.strip()
    keywords = [k.strip() for k in keywords_text.split(",")]
    return keywords


def fuzzy_search(query_keywords, documents, threshold=70):
    """
    Perform fuzzy search on documents using extracted keywords.

    Args:
        query_keywords: List of keywords extracted from the query
        documents: List of document objects with page_content field
        threshold: Minimum fuzzy match score to consider (0-100)

    Returns:
        List of documents sorted by relevance score
    """
    results = []

    for doc in tqdm(documents, desc="Fuzzy searching"):
        content = doc.page_content.lower()
        max_score = 0

        # Calculate the best match score for each keyword
        for keyword in query_keywords:
            keyword = keyword.lower()

            # Try exact match first (with higher weight)
            if keyword in content:
                score = 100
            else:
                # Try fuzzy matching
                score = fuzz.partial_ratio(keyword, content)

            max_score = max(max_score, score)

        if max_score >= threshold:
            results.append((doc, max_score))

    # Sort by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    # Return just the documents
    return [doc for doc, score in results]


def hybrid_retrieve_documents(query, vector_db=None, k=3, use_hybrid=True):
    """
    Retrieve documents using both vector similarity and keyword-based fuzzy search.

    Args:
        query: User query string
        vector_db: Vector database to search
        k: Number of documents to retrieve
        use_hybrid: Whether to use hybrid retrieval or just vector search

    Returns:
        List of retrieved documents
    """
    if vector_db is None:
        return []

    # Get vector-based results
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    vector_docs = retriever.invoke(query)

    if not use_hybrid:
        return vector_docs

    # Get all documents from the collection for fuzzy search
    # Note: This is a simplified approach. In production, you'd want to optimize this.
    all_docs = vector_db.get()["documents"]

    # Extract keywords using GPT-4o
    keywords = extract_keywords(query)

    # Perform fuzzy search
    fuzzy_docs = fuzzy_search(keywords, all_docs, threshold=70)[:k]

    # Combine results (removing duplicates)
    seen_contents = set()
    combined_docs = []

    # Add vector docs first (higher priority)
    for doc in vector_docs:
        if doc.page_content not in seen_contents:
            combined_docs.append(doc)
            seen_contents.add(doc.page_content)

    # Add fuzzy docs
    for doc in fuzzy_docs:
        if doc.page_content not in seen_contents and len(combined_docs) < k * 2:
            combined_docs.append(doc)
            seen_contents.add(doc.page_content)

    # Return top k combined results
    return combined_docs[:k]
