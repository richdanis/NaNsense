import gradio as gr
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompts import generate_answer, load_prompts
from keyword_retrieval import hybrid_retrieve_documents


# Load the vector database
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


def retrieve_documents(query, k=3, vector_db=None, use_hybrid=False):
    """
    Retrieve relevant documents from the vector database based on the query.

    Args:
        query: User query string
        k: Number of documents to retrieve
        vector_db: Vector database to search
        use_hybrid: Whether to use hybrid retrieval

    Returns:
        List of retrieved documents
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


def query_rag(query, k=3, vector_db=None, use_hybrid=False):
    """
    Process a query using RAG and return the answer.

    Args:
        query: User query
        k: Number of documents to retrieve
        vector_db: Vector database to search
        use_hybrid: Whether to use hybrid retrieval

    Returns:
        Generated answer based on retrieved context
    """
    # Retrieve relevant documents
    retrieved_docs = retrieve_documents(query, k, vector_db, use_hybrid)

    # Extract the page content from the documents
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    # Generate answer using the retrieved context
    response = generate_answer(
        query=query,
        retrieved_texts=retrieved_texts,
        prompt_template=prompt_template,
        model="gpt-4o-mini",
    )

    return response


# Create Gradio interface
def process_query(query, k, use_hybrid):
    if not query.strip():
        return "Please enter a query."

    try:
        return query_rag(query, k=k, vector_db=vector_db, use_hybrid=use_hybrid)
    except Exception as e:
        return f"An error occurred: {str(e)}"


# Launch the interface
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Ensure API key is set
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # Load prompts
    prompts = load_prompts()
    prompt_template = prompts["default"]

    # Load the vector database
    if os.path.exists("./chroma_db"):
        vector_db = load_vector_db()
    else:
        vector_db = None

    # Define the Gradio interface
    with gr.Blocks(title="NaNsense") as demo:
        gr.Markdown("# NaNsense - RAG-powered Question Answering")
        gr.Markdown("Ask questions about the documents in our knowledge base.")

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask something about the documents...",
                    lines=2,
                )
            with gr.Column(scale=1):
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Number of documents to retrieve",
                )
                hybrid_checkbox = gr.Checkbox(
                    label="Use Hybrid Retrieval (Keywords + Vectors)",
                    value=True,
                )

        submit_btn = gr.Button("Submit")

        answer_output = gr.Textbox(label="Answer", lines=10)

        submit_btn.click(
            fn=process_query,
            inputs=[query_input, k_slider, hybrid_checkbox],
            outputs=answer_output,
        )

        query_input.submit(
            fn=process_query,
            inputs=[query_input, k_slider, hybrid_checkbox],
            outputs=answer_output,
        )
    demo.launch()
