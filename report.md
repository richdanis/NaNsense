# NaNsense RAG Agent 🚀

## Data Processing Pipeline

### Data Cleaning

Our pipeline begins with thorough data cleaning to remove noise and significantly reduce the size of the dataset:

1. **URL Filtering**: We implemented intelligent filtering to retain only URLs containing meaningful content while excluding resource files (CSS, JavaScript, images).
2. **Text Processing**: Applied stopword removal to eliminate common words that don't contribute to semantic meaning.
3. **Content Trimming**: Removed the first 10% of text content from each page, as this often contains navigation elements and headers rather than substantive information.

This cleaning process reduced the size of the dataset from 17GB to 1.9GB.

### Document Chunking

To optimize retrieval performance, we:

1. Segmented documents into manageable chunks (500 characters with 100-character overlap)
2. Preserved metadata associations between chunks and their source URLs
3. Created a structured document format compatible with our vector database

This chunking strategy balances the granularity needed for precise retrieval with the context required for meaningful responses.

## Knowledge Base Creation

We implemented a vector database using Chroma with the following components:

1. **Embedding Model**: Utilized Hugging Face's `sentence-transformers/all-MiniLM-L6-v2` for generating semantic embeddings
2. **Storage**: Persisted the vector database to disk for efficient reuse
3. **Metadata Integration**: Maintained source URL information to provide attribution in responses

The resulting knowledge base contains thousands of document chunks, each with its semantic representation and source metadata.

## Retrieval System

Our retrieval system employs a similarity-based search approach:

1. **Query Embedding**: User queries are converted to the same vector space as our documents
2. **Semantic Search**: The system identifies the most semantically similar documents to the query
3. **Relevance Ranking**: Documents are ranked by similarity score, with the top-k returned

This approach allows us to retrieve contextually relevant information even when exact keyword matches aren't present.

## Response Generation

For generating responses, we:

1. **Context Assembly**: Combine retrieved documents into a comprehensive context
2. **Prompt Engineering**: Developed specialized prompts that instruct the model to use only the provided context
3. **LLM Integration**: Leveraged OpenAI's GPT-4o model to generate natural, accurate responses

Our prompt design emphasizes factual accuracy and attribution to the source material, reducing hallucination.

## Evaluation

To evaluate our system, we created a benchmark dataset consisting of 50 question-answer pairs using o1.
We prompted o1 to generate questions with single-word answers for randomly selected pages.
Evaluation is done by comparing the answer from the model with the answer from the ground truth and checking for a match.

## User Interface

We implemented a Gradio-based interface that allows users to submit natural language queries and view generated responses with source attribution.
Additionally, users can adjust the number of documents retrieved and the threshold for similarity score.
This interface makes the system accessible to non-technical users while providing transparency about the sources of information.

## Conclusion

Our RAG implementation demonstrates the effectiveness of combining semantic search with large language models for information retrieval and question answering. The system successfully leverages a large corpus of website content to provide accurate, contextually relevant responses to user queries.

Future improvements could include more sophisticated chunking strategies, hybrid retrieval methods combining semantic and keyword search, and more extensive evaluation using human feedback.