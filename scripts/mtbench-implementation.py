from openai import AsyncOpenAI
import json
import os
from fire import Fire
from tqdm.asyncio import tqdm as async_tqdm
import asyncio

mtbench_prompt_template = "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"


def get_requests(sample_dir: str) -> list:
    for file in os.listdir(sample_dir):
        if file.endswith(".json") and not file.startswith("single"):
            with open(os.path.join(sample_dir, file)) as f:
                item = json.load(f)
            question = item['data']['model_details']['role_messages']['candidate'][1:4]
            question_txt = '\n'.join([f"{q['role']}: {q['content']}" for q in question])
            answer_txt = item['data']['model_details']['role_messages']['candidate'][4]['content']
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": mtbench_prompt_template.format(question=question_txt, answer=answer_txt)}
            ]
            yield {
                "file": file,
                "messages": messages
            }

async def submit_worker(queue: asyncio.Queue, client: AsyncOpenAI, progress_bar: async_tqdm, output_dir: str):
    while True:
        request = await queue.get()
        if os.path.exists(os.path.join(output_dir, request['file'])):
            progress_bar.update(1)
            queue.task_done()
            continue
        try:
            response = await client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=request['messages'],
                max_tokens=400,
                n=1,
                temperature=0.0,
                seed=0,
            )
            response = response.model_dump()
            response = {
                'response': response["choices"][0]["message"]["content"]
            }
            with open(os.path.join(output_dir, request['file']), 'w') as f:
                json.dump(response, f, indent=2)
        except Exception as e:
            print(e)
        progress_bar.update(1)
        queue.task_done()


async def run(sample_dir, output_dir):
    clients = [AsyncOpenAI(
        api_key="<your-openai-api-key>",
        base_url="<your-base-url>",
        timeout=120,
        max_retries=3,
    ) for _ in range(8)]
    queue = asyncio.Queue()
    for request in get_requests(sample_dir):
        queue.put_nowait(request)
    progress_bar = async_tqdm(total=queue.qsize())
    tasks = [
        asyncio.create_task(submit_worker(queue, client, progress_bar, output_dir))
        for client in clients
    ]
    await queue.join()
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    progress_bar.close()


def main(sample_dir: str, output_dir: str):
    asyncio.run(run(sample_dir, output_dir))


if __name__ == '__main__':
    Fire(main)
