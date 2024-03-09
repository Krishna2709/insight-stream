import os

from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent, OpenAIAssistantAgent

from llama_index.readers.youtube_transcript.base import YoutubeTranscriptReader

from llama_index.core import download_loader
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core import ServiceContext
from llama_index.core import Prompt
from llama_index.core.prompts import PromptTemplate
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.vector_stores.deeplake import DeepLakeVectorStore

load_dotenv()


llm = OpenAI(model="gpt-4", temperature=0.5)
service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm)

# Youtube Video Transcript
loader = YoutubeTranscriptReader(is_remote=True)

youtube_docs = loader.load_data(ytlinks=['https://youtu.be/y9k-U9AuDeM?si=NbiP_pTAzLFzFYoE'])

yt_index = VectorStoreIndex.from_documents(youtube_docs, service_context=service_context)

yt_engine = yt_index.as_query_engine()