import gradio as gr
import os
from dotenv import load_dotenv
from prompts import load_prompts, generate_answer_streaming
from evaluate import load_vector_db, retrieve_documents


def query_rag_streaming(query, k=3, vector_db=None, use_hybrid=False):
    """
    Process a query using RAG and stream the answer.
    """
    retrieved_docs = retrieve_documents(query, k, vector_db, use_hybrid)
    retrieved_texts = [doc.page_content for doc in retrieved_docs]

    for response_chunk in generate_answer_streaming(
        query=query,
        retrieved_texts=retrieved_texts,
        prompt_template=prompt_template,
        model="gpt-4o-mini",
    ):
        yield response_chunk


def process_query_streaming(query, k, use_hybrid):
    if not query.strip():
        return "Please enter a query."

    try:
        full_response = ""
        for chunk in query_rag_streaming(
            query, k=k, vector_db=vector_db, use_hybrid=use_hybrid
        ):
            full_response += chunk
            yield full_response
    except Exception as e:
        yield f"An error occurred: {str(e)}"


if __name__ == "__main__":

    # so this interface allows to query the RAG agent and will output the answer in a stream
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

        answer_output = gr.Textbox(label="Answer", lines=10)

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

        with gr.Row():
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit")
            with gr.Column(scale=4):
                pass

        # add the buttons
        submit_btn.click(
            fn=process_query_streaming,
            inputs=[query_input, k_slider, hybrid_checkbox],
            outputs=answer_output,
            api_name="submit",
            queue=True,
        )

        query_input.submit(
            fn=process_query_streaming,
            inputs=[query_input, k_slider, hybrid_checkbox],
            outputs=answer_output,
            queue=True,
        )
    demo.launch()
