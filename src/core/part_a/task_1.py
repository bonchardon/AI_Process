# todo: Design  comprehensive prompt th instrcts n AI gent to hnde tsks specific to yor
#  chosen scenrio.

from os import environ
from typing import Any
from ast import literal_eval

import openai
from openai import OpenAI, AsyncOpenAI

from loguru import logger
from dotenv import load_dotenv
import numpy as np
from openai.types.chat import ChatCompletion

from sklearn.metrics.pairwise import cosine_similarity

from core.part_a.prompt_construct import PromptBuilder
from core.part_a.consts import OPEN_AI_EMBEDDINGS, OPEN_KEY_ASSISTANT, ASSISTANT_1_PATH, DOCUMENTS

load_dotenv()


class PromptTask:
    client: OpenAI = OpenAI(api_key=environ.get(OPEN_KEY_ASSISTANT))
    aclient: AsyncOpenAI = AsyncOpenAI(api_key=environ.get(OPEN_KEY_ASSISTANT))

    def __init__(self, message: str):
        self.message: str = message

    @staticmethod
    async def load_text_from_file(file_path: str) -> str | None:
        try:
            with open(file_path, 'r') as file:
                content: str = file.read()
            return content
        except Exception as e:
            logger.error(f'Error reading file: {e}')

    async def retrieve_documents(self, query: str) -> list[str]:
        documents: str = await self.load_text_from_file(DOCUMENTS)
        documents_list: list[str] = literal_eval(documents)

        document_embeddings = await self.get_embeddings(documents_list)
        query_embedding = await self.get_embeddings([query])

        document_embeddings = document_embeddings.reshape(len(documents_list), -1)
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, document_embeddings)
        return documents_list[similarities.argmax()]

    @staticmethod
    async def get_embeddings(texts: Any) -> np:
        response = openai.embeddings.create(model=OPEN_AI_EMBEDDINGS, input=texts)
        embeddings = [embedding.embedding for embedding in response.data]
        return np.array(embeddings)

    async def prompt_agent(self):
        relevant_document = await self.retrieve_documents(query=self.message)
        prompt_data = await PromptBuilder.prompt_data(message=self.message)

        if isinstance(prompt_data.messages, str):
            prompt_data.messages = [
                {'role': 'system', 'content': await self.load_text_from_file(ASSISTANT_1_PATH)},
                {'role': 'user', 'content': self.message},
                {'role': 'system', 'content': f'Relevant Documents:\n{relevant_document}'}
            ]

        try:
            response: ChatCompletion = await self.aclient.chat.completions.create(
                model=prompt_data.model,
                messages=prompt_data.messages,
                stream=prompt_data.stream
            )
            if response:
                choice: str | None = response.choices[0].message.content
                logger.info(choice)
                return choice

        except Exception as e:
            logger.error(f'Error during OpenAI request: {e}')
