import gradio as gr
import os
from dotenv import load_dotenv
from prompts import generate_answer, load_prompts
from evaluate import load_vector_db, retrieve_documents


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
    retrieved_docs = retrieve_documents(query, k, vector_db, use_hybrid)

    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    response = generate_answer(
        query=query,
        retrieved_texts=retrieved_texts,
        prompt_template=prompt_template,
        model="gpt-4o-mini",
    )

    return response


def process_query(query, k, use_hybrid):
    if not query.strip():
        return "Please enter a query."

    try:
        return query_rag(query, k=k, vector_db=vector_db, use_hybrid=use_hybrid)
    except Exception as e:
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    prompts = load_prompts()
    prompt_template = prompts["default"]

    if os.path.exists("./chroma_db"):
        vector_db = load_vector_db()
    else:
        vector_db = None

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
