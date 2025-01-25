from asyncio import run

from core.part_a.task_1 import PromptTask

async def main() -> None:
    await PromptTask(message='What should I do next?').prompt_agent()

if __name__ == '__main__':
    run(main())
