# CatholicRAG

A chatbot that answers questions about Catholic teaching using only the Bible (NABRE) and the Catechism of the Catholic Church, with a citation for every claim.

## How it works
1. **Ingest:** each Catechism paragraph and Bible verse is embedded and stored in Pinecone.
2. **Retrieve:** the question is embedded and the 5 closest passages are pulled.
3. **Answer:** GPT-4o-mini answers using only those passages and cites them inline, like [CCC 1213] or [John 3:16]. If nothing relevant is found, it says so instead of guessing.

## Tech
Python, FastAPI, OpenAI API, Pinecone

## Run it
1. `pip install fastapi uvicorn openai pinecone python-dotenv`
2. Add `OPENAI_API_KEY` and `PINECONE_API_KEY` to a `.env` file
3. `python ingest_data.py` (one time, builds the index)
4. `uvicorn app:app --reload` and open http://127.0.0.1:8000
