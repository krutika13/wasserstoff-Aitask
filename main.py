from fastapi import FastAPI
from queue import Queue
from fastapi import FastAPI
import asyncio
from fastapi.responses import StreamingResponse
# from load_model import load_model
from threading import Thread
import time
from queue import Queue
from auto_gptq import AutoGPTQForCausalLM
from pydantic import BaseModel, BaseSettings
from transformers import AutoTokenizer, TextStreamer, pipeline, TextIteratorStreamer
import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from fastapi import BackgroundTasks

from threading import Thread
from typing import Optional

from langchain import PromptTemplate, LLMChain
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, ConversationalRetrievalChain

# creating a fast application
app = FastAPI()
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEFAULT_SYSTEM_PROMPT = """ You are a helpful, respectful and honest Assistant. Please ensure that your responses are socially unbiased and positive in nature.
Don't give answer extra, only give response point to point the question which is asked.
""".strip()


def generate_prompt(prompt: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
[INST] <>
{system_prompt}
<>

{prompt.strip()} [/INST]
""".strip()


def initialize_model_and_tokenizer(model_name="TheBloke/Llama-2-7B-chat-GPTQ"):
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        revision="gptq-4bit-128g-actorder_True",
        model_basename="model",
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    return model, tokenizer


def init_chain(model, tokenizer):
    class CustomLLM(LLM):
        """Streamer Object"""

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = inputs.to(DEVICE)
            kwargs = dict(input_ids=inputs["input_ids"], streamer=self.streamer, max_new_tokens=1000)
            thread = Thread(target=model.generate, kwargs=kwargs)
            thread.start()
            return ""

        @property
        def _llm_type(self) -> str:
            return "custom"

    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-base", model_kwargs={"device": DEVICE},
        encode_kwargs={'normalize_embeddings': False}
    )
    print("embedding done!")
    # persist_directory = "db"

    import requests
    from bs4 import BeautifulSoup

    url = 'https://ai6642.wordpress.com/2024/06/22/the-current-revolution-in-generative-ai-transforming-industries-and-innovation/'
    # url="https://en.wikipedia.org/wiki/Artificial_intelligence"
    # Make a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title and content
        title_element = soup.find('h1', class_='entry-title')
        if title_element:
            title = title_element.text.strip()
        else:
            title = "Title not found"
        
        content_element = soup.find('div', class_='entry-content')
        if content_element:
            content = content_element.text.strip()
        else:
            content = "Content not found"
        
        print(f"Title: {title}")
        print(f"Content:\n{content}")
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")

    metadata = {
    'source': 'https://ai6642.wordpress.com/2024/06/22/the-current-revolution-in-generative-ai-transforming-industries-and-innovation/',
    'retrieved_at': '2024-06-24'
    }

    text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)

    chunks = text_splitter.split_text(content)
    documents = [Document(page_content=chunk, metadata=metadata) for chunk in chunks]

    for doc in documents:
        print(f"Chunk: {doc.page_content}\nMetadata: {doc.metadata}\n") 
        db = Chroma.from_documents(documents, embeddings,persist_directory = "MyDB")
        db.persist()
    torch.cuda.empty_cache()
    llm = CustomLLM()
    torch.cuda.empty_cache()

    # template = """Question: {question}
    # Answer: Let's think step by step."""
    # prompt = PromptTemplate(template=template, input_variables=["question"])
    template = generate_prompt(
        """
    {context}

    Question: {question}
    """,
        # system_prompt=SYSTEM_PROMPT,
    )
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    retriever = vectordb.as_retriever(search_type='mmr', search_kwargs={'k': 8})
    llm_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
        chain_type="stuff",
        get_chat_history=lambda h: h,
        # chain_type_kwargs={"prompt": prompt},
        combine_docs_chain_kwargs={'prompt': prompt},
        max_tokens_limit=1500,
    )
    # llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain, llm


model, tokenizer = initialize_model_and_tokenizer()
print("Model Init Done")
llm_chain, llm = init_chain(model, tokenizer)
print("LLM Chain Done and LLM Done")


async def serve_data(message):
    # history = [[message,None]]
    print("calling now")
    llm_chain({'question': message, 'chat_history': []})
    print("calling done")
    for character in llm.streamer:
        print(character)
        # history += character
        yield str(character)
    await asyncio.sleep(0.5)





@app.get("/stream")
async def stream_data():
    # Function to generate streamed data
    async def generate():
        print("calling now")
        message = "What is Generative AI?"
        llm_chain({'question': message, 'chat_history': []})
        print("calling done")
        for character in llm.streamer:
            # for i in range(10):
            print(character)
            yield f"{character}"
            await asyncio.sleep(0.01)  # Simulate some async operation

    return StreamingResponse(generate(), media_type="text/event-stream")
