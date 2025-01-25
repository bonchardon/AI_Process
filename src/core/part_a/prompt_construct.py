from venv import logger

from core.part_a.models import PromptAI
from core.part_a.consts import GPT_3_5_TURBO


class PromptBuilder:
    @classmethod
    async def prompt_data(cls, message: str) -> PromptAI | None:
        if not (prompt_constructor := PromptAI(
            model=GPT_3_5_TURBO,
            messages=message,
            stream=False
        )):
            logger.error('Something went wrong when trying to construct a prompt. ')
            return
        logger.info('Well done! Prompt is working just fine.')
        return prompt_constructor
