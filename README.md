**Navigation through the repository files:**
- Creating the vector store using DeepLake: `creating_arxiv_dataset_vector_store.ipynb`
- Insight Stream main code: `server/src/insight.py`
- FastAPI Endpoints: `server/src/app.py`
- Docker File to deploy endpoints: `server/Dockerfile`
- Streamlit web application: `static/app.py`

# Project Overview *(basic version)*

- **Goal**: This is an insight bot for YouTube videos. Its goal is to summarize, generate questions, and obtain relevant information from the RAG.
- **Motivation**: Whenever I attend webinars, I always have questions on specific topics, and sometimes, I don't even know what questions I could ask. I have always thought of building a bot that can listen to webinars and generate questions along with relevant content.
- **Dataset**: arXiv Dataset (https://www.kaggle.com/datasets/Cornell-University/arxiv/data)
- **Technology Stact**:
  ```
    Python
  
    LLamaIndex (to build the RAG system)
  
    DeepLake (vector database)
  
    FastAPI (to create and expose the endpoints)
  
    OpenAI (for LLMs)
  
    Google Cloud Run (server to deploy the FastAPI endpoints)
  
    Streamlit (for front-end)
  ```
- **Working**: In this basic version, whenever the user enters a YouTube link:
    1. The app sends an API call to the FastAPI server with the YouTube link as the payload.
    2. The vector database (arXiv dataset) is loaded. An index is created from this vector store.
    3. Using the `YoutubeTranscriptReader`, the video is loaded, transcribed, and indexed.
    4. This index is initiated as a *query engine* to communicate with the video transcript.
    5. This *query engine* is used to get the video summary
    ```
        loader = YoutubeTranscriptReader(is_remote=True)
    
        youtube_docs = loader.load_data(ytlinks=[youtube_url])
    
        youtube_index = VectorStoreIndex.from_documents(youtube_docs, service_context=self.service_context)
    
        youtube_engine = youtube_index.as_query_engine(output_cls=<Output Parser>)
    
        youtube_summarize = youtube_engine.query("Summarize the video and generate technical questions for the video to ask the speaker.")
    ```
    6. The vector database index is also initiated as a *query engine* and queried to extract relevant papers based on the summary.
    ```
        db = DeepLakeVectorStore(
            dataset_path=<dataset_url>,
            read_only=True,
            overwrite=False
        )
    
        arxiv_index = VectorStoreIndex.from_vector_store(self.db, service_context=self.service_context)

        # RAG Vector Engine
        vector_engine = arxiv_index.as_query_engine(output_cls=<Output Parser>, similarity_top_k=5)
    
        query = f"Extract research papers relevant to the video summary: {youtube_summarize.summary}"
    
        papers = vector_engine.query(query)
    ```
   7. The summary, questions, and relevant papers are displayed to the user.
   8. Now, a chat interface appears where the user can interact with the RAG system.
   9. When the user asks a question, an API is called along with the prompt payload to get the response and relevant papers to the response.
  
### Application Architecture
----
![Insight Stream Architecture](https://github.com/Krishna2709/home-assignment-project/blob/master/images/InsightStream_Arch.png)

### Web application Design
----
![Web app design](https://github.com/Krishna2709/home-assignment-project/blob/master/images/InsightStream_Design.png)

### Steps to interact with the application and endpoints 
----
**_Make sure to interact with the Analyzer Endpoint before Query Endpoint_**
1. FastAPI Endpoints: 
```
https://insight-stream-service-kvbcxmn5bq-uc.a.run.app/docs
```
2. Webapp (https://github.com/Krishna2709/insight-stream-webapp): 
```
https://insight-stream.streamlit.app
```
3. (Optional) Using Postman for API interaction (⚠️ will update soon)
4. Examples
![](https://github.com/Krishna2709/home-assignment-project/blob/master/images/Example1.png)
![](https://github.com/Krishna2709/home-assignment-project/blob/master/images/Example1_2.png)
![](https://github.com/Krishna2709/home-assignment-project/blob/master/images/Example1_3.png)
![](https://github.com/Krishna2709/home-assignment-project/blob/master/images/Example2_1.png)
![](https://github.com/Krishna2709/home-assignment-project/blob/master/images/Example2_2.png)
![](https://github.com/Krishna2709/home-assignment-project/blob/master/images/Example2_3.png)
![](https://github.com/Krishna2709/home-assignment-project/blob/master/images/Example2_4.png)
### Neglected Improvements due to time constraints
----
⚠️ *Will update the Question Generation prompt in some time since the prompt is not clear. The model generates questions from the video transcription(like quizzes) instead of questions for the speaker.*

- `gpt-3.5-turbo` to reduce latency or response time
- Prompt Engineering - for effective responses.
- UI/UX optimization
- Chat functionality with memory ability
- Logging for evaluation
- Improving retrieval methods like Rerank wrapper
- LLM variants
- Asynchronous methods

### Further Improvements
----

- Chat ability using Chat Engine, not just Query Engine
- Prompt Engineering
- RAG and Chat evaluations. Logging
- Reducing latency
- Live video transcription and RAG querying.
- Using Bark(Suno's open-source text-to-speech model) to ask questions


### Resources
----

- activeloop RAG for Production (https://learn.activeloop.ai/courses/rag)
- LLamaIndex Documentation (https://docs.llamaindex.ai/en/stable/)
- LLamaIndex YouTube Channel (https://www.youtube.com/@LlamaIndex)
