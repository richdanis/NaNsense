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
    print(retrieved_texts)
    for response_chunk in generate_answer_streaming(
        query=query,
        retrieved_texts=retrieved_texts,
        prompt_template=prompt_template,
        model="gpt-4o",
    ):
        yield response_chunk


def process_conversation_streaming(query, conversation_history):
    """
    Process a follow-up query using the conversation history as context.
    """
    if not query.strip():
        return conversation_history

    try:
        full_response = conversation_history
        if full_response:
            full_response += f"\n\nYou: {query}\n\nAssistant: "
        else:
            full_response = f"You: {query}\n\nAssistant: "

        assistant_response = ""

        for chunk in generate_answer_streaming(
            query=full_response,
            retrieved_texts=[],  # No RAG retrieval for follow-ups
            prompt_template=prompts["conversation_follow_up"],
            model="gpt-4o",
        ):
            assistant_response += chunk
            yield full_response + assistant_response
    except Exception as e:
        yield f"{full_response}An error occurred: {str(e)}"


def process_query_streaming(query, k, use_hybrid, conversation_history):
    if not query.strip():
        return conversation_history

    if conversation_history.strip():
        yield from process_conversation_streaming(query, conversation_history)
        return

    try:
        full_response = f"You: {query}\n\nAssistant: "
        assistant_response = ""

        for chunk in query_rag_streaming(
            query, k=k, vector_db=vector_db, use_hybrid=use_hybrid
        ):
            assistant_response += chunk
            yield full_response + assistant_response
    except Exception as e:
        yield f"{full_response}An error occurred: {str(e)}"


if __name__ == "__main__":
    load_dotenv()

    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    prompts = load_prompts()
    prompt_template = prompts["rag_default"]

    vector_db = None

    if os.path.exists("chroma_db"):
        vector_db = load_vector_db()
        print("Vector database loaded")
    else:
        vector_db = None

    with gr.Blocks(title="NaNsense") as demo:
        gr.Markdown("# NaNsense - RAG-powered Question Answering")
        gr.Markdown("Ask questions about the documents in our knowledge base.")

        conversation_output = gr.Textbox(label="Conversation", lines=15)

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
                    value=False,
                )
                clear_btn = gr.Button("Clear Conversation")

        with gr.Row():
            with gr.Column(scale=1):
                submit_btn = gr.Button("Submit")
            with gr.Column(scale=4):
                pass

        def clear_conversation():
            return ""

        submit_btn.click(
            fn=process_query_streaming,
            inputs=[query_input, k_slider, hybrid_checkbox, conversation_output],
            outputs=conversation_output,
            api_name="submit",
            queue=True,
        )

        query_input.submit(
            fn=process_query_streaming,
            inputs=[query_input, k_slider, hybrid_checkbox, conversation_output],
            outputs=conversation_output,
            queue=True,
        )

        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=conversation_output,
            queue=False,
        )

    demo.launch()
