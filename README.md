# Penn Course LLM
A RAG-based LLM with access to more than 10,000 courses within Penn Courses Catalog to help Penn students choose their courses more quickly and easily!


## How to run:

1. Create a .env file to use OpenAI and Pinecone API services with the following format

```
OPENAI_API_KEY=<OPENAI API KEY>
PINECONE_API_KEY=<PINECONE API KEY>
```

2. Run the vector database creation script to create a database of Penn Courses Catalog data for retrieval augmentation for the LLM

```
python data/vector_db.py
```

3. Run the application to interact with the RAG-based LLM

```
python model/gpt-4.py
```

A demo can be found here: https://drive.google.com/file/d/1x04-kCscRWlvGLDrMBN7YAB4GmFYNyxa/view?usp=sharing
