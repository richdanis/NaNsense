import json
import os
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompts import generate_answer, load_prompts


def load_vector_db(persist_directory="./chroma_db"):
    """
    Load the existing Chroma vector database.

    Args:
        persist_directory: Directory where the vector database is stored

    Returns:
        Loaded Chroma vector database
    """
    # Initialize the embedding model - same as used for creating the DB
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    # Load the existing vector database
    vectordb = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    return vectordb


def retrieve_documents(query, k=3, vector_db=None):
    """
    Retrieve relevant documents from the vector database based on the query.

    Args:
        query: User query string
        k: Number of documents to retrieve
        vector_db: Vector database to search

    Returns:
        List of retrieved documents
    """
    if vector_db is None:
        return []
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(query)
    return docs


def check_exact_match(prediction, reference):
    """
    Check if the prediction exactly matches the reference.

    Args:
        prediction: Predicted answer
        reference: Reference answer

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    # Normalize both strings by removing extra whitespace and converting to lowercase
    prediction_norm = " ".join(prediction.lower().split())
    reference_norm = " ".join(reference.lower().split())

    # Check for exact match
    return 1.0 if prediction_norm == reference_norm else 0.0


def evaluate_rag_system(
    benchmark_file, k=3, prompt_template_name="rag_default", model="gpt-4o-mini"
):
    """
    Evaluate the RAG system using a benchmark dataset.

    Args:
        benchmark_file: Path to the benchmark JSON file
        k: Number of documents to retrieve for each query
        prompt_template_name: Name of the prompt template to use
        model: Model to use for generation

    Returns:
        Dictionary with evaluation metrics
    """
    # Load the benchmark data
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    # Load the vector database
    vector_db = load_vector_db()

    # Load prompts
    prompts = load_prompts()
    prompt_template = prompts[prompt_template_name]

    results = []

    for item in tqdm(benchmark_data, desc="Evaluating"):
        question = item["question"]
        reference_answer = item["answer"]

        # Retrieve documents
        retrieved_docs = retrieve_documents(question, k=k, vector_db=vector_db)
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        # Generate answer
        predicted_answer = generate_answer(
            query=question,
            retrieved_texts=retrieved_texts,
            prompt_template=prompt_template,
            model=model,
        )

        # Check for exact match
        match_score = check_exact_match(predicted_answer, reference_answer)

        # Store result
        result = {
            "question": question,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "exact_match": match_score,
            "source_url": item.get("page_url", "Unknown"),
            "website": item.get("website", "Unknown"),
        }

        results.append(result)

    # Calculate metrics
    match_scores = [r["exact_match"] for r in results]
    metrics = {
        "exact_match_accuracy": np.mean(match_scores),
        "num_exact_matches": sum(match_scores),
        "total_samples": len(results),
        "exact_match_percentage": (
            (sum(match_scores) / len(results)) * 100 if results else 0
        ),
    }

    return {"metrics": metrics, "results": results}


def save_evaluation_results(results, output_file="evaluation_results.json"):
    """
    Save evaluation results to a JSON file.

    Args:
        results: Evaluation results dictionary
        output_file: Path to save the results

    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {output_file}")


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Ensure API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Set parameters
    benchmark_file = "benchmark.json"
    k = 3  # Number of documents to retrieve
    prompt_template_name = "rag_default"
    model = "gpt-4o-mini"

    # Run evaluation
    results = evaluate_rag_system(
        benchmark_file=benchmark_file,
        k=k,
        prompt_template_name=prompt_template_name,
        model=model,
    )

    # Print summary metrics
    print("\nEvaluation Results:")
    for metric, value in results["metrics"].items():
        print(
            f"{metric}: {value:.2f}%"
            if "percentage" in metric
            else f"{metric}: {value}"
        )

    # Save results
    save_evaluation_results(results)
