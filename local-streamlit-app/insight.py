from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.readers.youtube_transcript.base import YoutubeTranscriptReader
from llama_index.core import VectorStoreIndex
from llama_index.core import ServiceContext
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.deeplake import DeepLakeVectorStore

from pydantic import BaseModel, Field
from typing import List

load_dotenv()

# =========== Parsers =========== #

class YoutubeVideoTranscriptSummary(BaseModel):
    """Data model for a call summary."""
    summary: str = Field(
        description="Summary of the youtube video transcript. Should not exceed 6 sentences."
    )
    questions: List[str] = Field(
        description="List of generated technical questions"
    )

class Paper(BaseModel):
    """Data model for a research paper."""
    title: str = Field(description="The title of the research paper")
    abstract: str = Field(description="The abstract of the research paper")

class ResearchPapersWithTitlesAndAbstracts(BaseModel):
    """Data model for list of research papers."""
    papers: List[Paper] = Field(
        description="List of research papers, each with a title and abstract"
    )

# =========== Video Analyzer =========== #
class VideoAnalysis:
    def __init__(self):
        self.llm = OpenAI(model= "gpt-4", temperature=0.2)
        self.service_context = ServiceContext.from_defaults(chunk_size=1024, llm=self.llm)
        self.db = DeepLakeVectorStore(
            dataset_path="hub://blackorder/arxiv_docs_2024_03_08",
            read_only=True,
            overwrite=False
        )
        self.arxiv_index = VectorStoreIndex.from_vector_store(self.db, service_context=self.service_context)
        self.youtube_engine = None
        self.vector_engine = None

    def analyze(self, youtube_url: str):
        loader = YoutubeTranscriptReader(is_remote=True)
        youtube_docs = loader.load_data(ytlinks=[youtube_url])
        youtube_index = VectorStoreIndex.from_documents(youtube_docs, service_context=self.service_context)
        self.youtube_engine = youtube_index.as_query_engine(output_cls=YoutubeVideoTranscriptSummary)
        youtube_summarize = self.youtube_engine.query("Summarize the video")

        self.vector_engine = self.arxiv_index.as_query_engine(output_cls=ResearchPapersWithTitlesAndAbstracts)
        query = f"Extract research papers relevant to the video summary: {youtube_summarize.summary}"
        papers = self.vector_engine.query(query)

        return {
            "summary": youtube_summarize.summary,
            "questions": youtube_summarize.questions,
            "papers": [{"title": paper.title, "abstract": paper.abstract} for paper in papers.papers]
        }

# =========== Chatbot =========== #
def chat_engine(prompt, youtube_engine, vector_engine, llm):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=youtube_engine,
            metadata=ToolMetadata(
                name="youtube_transcript",
                description="Contains the YouTube video transcript useful for answering user queries."
            ),
        ),
        QueryEngineTool(
            query_engine=vector_engine,
            metadata=ToolMetadata(
                name="research_papers_with_title_and_abstract",
                description="Provides titles and abstracts of relevant research papers for each user query."
            ),
        ),
    ]
    
    agent = OpenAIAgent.from_tools(query_engine_tools, verbose=False, llm=llm)
    response = agent.chat(prompt)
    response = str(response)

    return response

