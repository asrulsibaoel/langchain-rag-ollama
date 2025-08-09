from pydantic import BaseModel
from typing import Dict, List, Literal, Optional

# Request body schema


class EmbedRequest(BaseModel):
    texts: List[str]
    metadatas: List[Dict[str, str]]

# Response schema (optional, improves OpenAPI docs)


class EmbedResponse(BaseModel):
    texts: List[str]
    ids: List[str]
    categories: Optional[List[str]]
    status: str


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: List[ChatTurn] = []

# class ChatResponse(BaseModel):
#     answer: BaseMessage
