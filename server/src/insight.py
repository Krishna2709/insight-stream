from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.readers.youtube_transcript.base import YoutubeTranscriptReader
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.prompts import PromptTemplate

from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# =========== Parsers =========== #

class YoutubeVideoTranscriptSummary(BaseModel):
    """Data model for a call summary."""
    summary: str = Field(
        description="A concise summary of the YouTube video transcript, not exceeding six sentences, serving as the basis for question generation."
    )
    questions: List[str] = Field(
        description="A curated list of contextually relevant questions intended for the speaker, derived from the video summary to simulate attendee inquiries or clarify key points discussed within the video content."
    )

class Paper(BaseModel):
    """Data model for a research paper."""
    title: str = Field(description="The title of the research paper")
    abstract: str = Field(description="The abstract of the research paper")

class ResearchPapersWithTitlesAndAbstracts(BaseModel):
    """Data model for list of research papers."""
    response: str = Field(
         description="The response to the user query limited to 3 sentences, along with the relevant research papers."
    )
    papers: List[Paper] = Field(
        description="List of research papers, each with a title and abstract"
    )


# =========== Video Analyzer =========== #
    
class VideoAnalysis:

    '''
    Analyzes the youtube video transcript and returns a summary, 
    technical questions, and relevant research papers.
    '''
    def __init__(self):
        self.llm = OpenAI(model= "gpt-4-turbo-preview", temperature=0.7)
        self.service_context = ServiceContext.from_defaults(chunk_size=1024, llm=self.llm) # Should be replaced with Settings in future
        # Loading the vector database
        self.db = DeepLakeVectorStore(
            dataset_path="hub://blackorder/arxiv_docs_2024_03_08",
            read_only=True,
            overwrite=False
        )
        self.arxiv_index = VectorStoreIndex.from_vector_store(self.db, service_context=self.service_context)
        self.youtube_engine = None
        self.vector_engine = None


    def analyze(self, youtube_url: str):

        '''
        Analyzes the youtube video transcript and returns a summary, 
        technical questions, and relevant research papers.

        Args: youtube_url

        Returns: summary, questions, and relevant papers
        '''

        # Youtube Vector Engine
        loader = YoutubeTranscriptReader(is_remote=True)
        youtube_docs = loader.load_data(ytlinks=[youtube_url])
        youtube_index = VectorStoreIndex.from_documents(youtube_docs, service_context=self.service_context)
        self.youtube_engine = youtube_index.as_query_engine(output_cls=YoutubeVideoTranscriptSummary)
        youtube_summarize = self.youtube_engine.query("Summarize the video and generate technical questions for the video to ask the speaker.")

        # RAG Vector Engine
        self.vector_engine = self.arxiv_index.as_query_engine(output_cls=ResearchPapersWithTitlesAndAbstracts, similarity_top_k=5)
        query = f"Extract research papers relevant to the video summary: {youtube_summarize.summary}"
        papers = self.vector_engine.query(query)


        return {
            "summary": youtube_summarize.summary,
            "questions": youtube_summarize.questions,
            "papers": [{"title": paper.title, "abstract": paper.abstract} for paper in papers.papers]
        }


# =========== Arxiv Data Query Engine =========== #
def query_engine(prompt, vector_engine):

    '''
    Queries the vector engine to get relevant research papers based on the user's prompt.

    Args: user prompt, vector engine

    Returns: response text and relevant papers
    '''

    # Prompt Template for the vector engine
    template = """ 
        Context information is below.
        ---------
        {context_str}
        ---------
        You are an intelligent research chatbot that can answer questions.
        Answer the user query along with the relevant research papers with title and abstract from the above context in JSON format, if exists.

        Query: {query_str}
        Answer: \
    """

    prompt_template = PromptTemplate(
        template=template
    )

    # Update the prompts of the vector engine: vector_engine.get_prompts()
    vector_engine.update_prompts(
        {"response_synthesizer:text_qa_template": prompt_template}
    )

    # Query the vector engine
    response = vector_engine.query(prompt)
    response = response.response

    # Extract the response text and relevant papers
    response_text = response.response
    papers = response.papers

    return {
        "response": response_text,
        "papers": papers
    }

