import json
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from prompts import generate_answer, load_prompts
from keyword_retrieval import hybrid_retrieve_documents


def load_vector_db(persist_directory="./chroma_db"):
    """
    Load the existing Chroma vector database.
    """
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    vectordb = Chroma(
        persist_directory=persist_directory, embedding_function=embeddings
    )

    return vectordb


def retrieve_documents(query, k=3, vector_db=None, use_hybrid=False):
    """
    Retrieve relevant documents from the vector database based on the query.
    """
    if vector_db is None:
        return []

    if use_hybrid:
        return hybrid_retrieve_documents(query, vector_db, k)
    else:
        retriever = vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        docs = retriever.invoke(query)
        return docs


def check_exact_match(prediction, reference):
    """
    Check if the prediction exactly matches the reference.
    """
    prediction_norm = " ".join(prediction.lower().split())
    reference_norm = " ".join(reference.lower().split())

    return 1.0 if prediction_norm == reference_norm else 0.0


def evaluate_rag_system(
    benchmark_file,
    k=3,
    prompt_template_name="rag_default",
    model="gpt-4o",
    use_hybrid=False,
):
    """
    Evaluate the RAG system using a benchmark dataset.
    """
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    vector_db = load_vector_db()

    prompts = load_prompts()
    prompt_template = prompts[prompt_template_name]

    results = []

    for item in tqdm(benchmark_data, desc="Evaluating"):
        question = item["question"]
        reference_answer = item["answer"]

        retrieved_docs = retrieve_documents(
            question, k=k, vector_db=vector_db, use_hybrid=use_hybrid
        )
        retrieved_texts = [doc.page_content for doc in retrieved_docs]

        predicted_answer = generate_answer(
            query=question,
            retrieved_texts=retrieved_texts,
            prompt_template=prompt_template,
            model=model,
        )

        match_score = check_exact_match(predicted_answer, reference_answer)

        result = {
            "question": question,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "exact_match": match_score,
            "source_url": item.get("page_url", "Unknown"),
            "website": item.get("website", "Unknown"),
        }

        results.append(result)

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


if __name__ == "__main__":
    load_dotenv()

    benchmark_file = "benchmark.json"
    k = 3
    prompt_template_name = "rag_default"
    model = "gpt-4o"
    use_hybrid = False

    # run evaluation
    results = evaluate_rag_system(
        benchmark_file=benchmark_file,
        k=k,
        prompt_template_name=prompt_template_name,
        model=model,
        use_hybrid=use_hybrid,
    )

    print("\nEvaluation Results:")
    for metric, value in results["metrics"].items():
        print(
            f"{metric}: {value:.2f}%"
            if "percentage" in metric
            else f"{metric}: {value}"
        )

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
