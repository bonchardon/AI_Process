from pydantic import BaseModel


class PromptAI(BaseModel):
    model: str
    messages: str
    stream: bool
