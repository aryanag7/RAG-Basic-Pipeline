# System Design RAG

A work-in-progress Retrieval-Augmented Generation application that answers
system-design questions from a PDF document.

## Current pipeline

- Loads and cleans a PDF
- Splits text into overlapping chunks
- Creates local Hugging Face embeddings
- Stores embeddings persistently in Chroma
- Retrieves the top five relevant chunks
- Filters weak matches using a distance threshold
- Generates grounded answers using an OpenAI model

The current implementation is intentionally kept in one `app.py` file.
It will be modularised in a later iteration.