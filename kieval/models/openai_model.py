import asyncio
import json
import os
import argparse
import logging
import time
import codecs
import traceback
import httpx

from typing import Optional, List, Dict, Union, Any
from random import choice
from tqdm.asyncio import tqdm as atqdm
from openai import AsyncOpenAI, APIStatusError


class OpenAIClient:
    def __init__(
        self,
        output_path: str,
        openai_model: str,
        openai_key: Union[str, List[str]],
        openai_api_base: Optional[Union[str, List[str]]] = None,
        openai_proxy: Optional[Union[str, List[str]]] = None,
        openai_timeout: Optional[float] = 60.0,
        openai_max_retries: Optional[int] = 5,
        generation_config: Optional[Dict] = None,
        max_error_count: Optional[int] = 100,
        trial_run=False,
        dump_individual_rsp=True,
    ):
        self.output_path = output_path
        self.trial_run = trial_run
        self.max_error_count = max_error_count
        self.total_errors = 0
        self.logger = logging.getLogger(__name__)

        if isinstance(openai_key, str):
            openai_key = [openai_key]

        if openai_api_base is None:
            openai_api_base = ["https://api.openai.com/v1"] * len(openai_key)
        elif isinstance(openai_api_base, str):
            openai_api_base = [openai_api_base] * len(openai_key)

        assert len(openai_key) == len(
            openai_api_base
        ), "Number of openai_key and openai_api_base must match"

        if openai_proxy is not None:
            if isinstance(openai_proxy, str):
                openai_proxy = [openai_proxy] * len(openai_key)
            else:
                assert len(openai_proxy) == len(
                    openai_key
                ), "Number of openai_key and openai_proxy must match"
            self.clients = [
                AsyncOpenAI(
                    api_key=key,
                    base_url=api_base,
                    timeout=openai_timeout,
                    max_retries=openai_max_retries,
                    http_client=httpx.AsyncClient(proxies=proxy),
                )
                for key, api_base, proxy in zip(
                    openai_key, openai_api_base, openai_proxy
                )
            ]
        else:
            self.clients = [
                AsyncOpenAI(
                    api_key=key,
                    base_url=api_base,
                    timeout=openai_timeout,
                    max_retries=openai_max_retries,
                )
                for key, api_base in zip(openai_key, openai_api_base)
            ]

        self.model = openai_model

        self.response_queue = asyncio.Queue()
        self.dump_individual_rsp = dump_individual_rsp

        if generation_config is None:
            self.generation_config = {
                "frequency_penalty": 0,
                "max_tokens": 100,
                "n": 1,
                "presence_penalty": 0,
                "response_format": {"type": "text"},
                "seed": 42,
                "stream": False,
                "temperature": 0.0,
            }
        else:
            self.generation_config = generation_config

        if dump_individual_rsp:
            os.makedirs(os.path.join(self.output_path, "responses"), exist_ok=True)

    async def query(
        self,
        request,
        num_retries=3,
    ):
        if isinstance(request, dict):
            request_dict = request
        else:
            request_dict = request.__dict__

        if "messages" not in request_dict and "prompt" in request_dict:
            request_dict["messages"] = [{"role": "user", "content": request_dict["prompt"]}]

        assert "messages" in request_dict, "messages must be provided in request"
        assert "uuid" in request_dict, "uuid must be provided in request"

        if self.dump_individual_rsp:
            save_path = os.path.join(
                self.output_path, "responses", f'{request_dict["uuid"]}.json'
            )

            if os.path.exists(save_path) and not self.trial_run:
                with codecs.open(save_path) as f:
                    rsp_content = json.load(f)
                await self.response_queue.put(rsp_content)
                return f"Skipping {save_path}"

        if "generation_config" in request_dict and isinstance(
            request_dict["generation_config"], dict
        ):
            generation_config = self.generation_config.copy()
            generation_config.update(request_dict["generation_config"])
        else:
            generation_config = self.generation_config

        response = None
        while num_retries > 0:
            num_retries -= 1
            try:
                client = choice(self.clients)
                response = await client.chat.completions.create(
                    messages=request_dict["messages"],
                    model=self.model,
                    **generation_config,
                )
                response = response.model_dump()
                break
            except APIStatusError as e:
                if self.max_error_count > self.total_errors:
                    self.total_errors += 1
                    self.logger.warning(
                        f"OpenAI APIStatusError: {e}, total errors: {self.total_errors}, sleeping..."
                    )
                    await asyncio.sleep(1.0)
                else:
                    self.logger.error(
                        f"OpenAI APIStatusError: {e}, max_error_count reached, exiting..."
                    )
                    raise e
            except:
                print(traceback.format_exc())

        if response is None:
            raise Exception("Empty response from remote OpenAI API")

        try:
            response["generated_text"] = response["choices"][0]["message"]["content"]
        except:
            print(traceback.format_exc())
            print(response)
            raise Exception("Empty response from remote OpenAI API")

        if self.dump_individual_rsp:
            with codecs.open(save_path, "w") as f:
                json.dump(
                    {"request": request_dict, "response": response},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        await self.response_queue.put({"request": request_dict, "response": response})

        return response["choices"][0]["message"]["content"]

    async def write_responses_to_file(self):
        save_path = os.path.join(self.output_path, "all_responses.jsonl")
        while True:
            response = await self.response_queue.get()
            with codecs.open(save_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(response, ensure_ascii=False) + "\n")
            self.response_queue.task_done()


class AsyncRateLimitThreadPool:
    def __init__(self, num_workers, num_requests, period):
        self.num_workers = num_workers
        self.num_requests = num_requests
        self.loop = asyncio.get_event_loop()
        self.semaphore = asyncio.Semaphore(num_workers)
        self.last_call_time = time.time()
        self.call_count = 0
        self.period = period

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def _rate_limited_call(self, func, *args, **kwargs):
        # Limit the number of calls to func per minute
        elapsed_time = time.time() - self.last_call_time
        if elapsed_time < self.period:
            self.call_count += 1
            if self.call_count > self.num_requests:
                sleep_time = self.period - elapsed_time
                # logging.info("Sleeping for {} seconds".format(sleep_time))
                await asyncio.sleep(sleep_time)
                self.call_count = 0
                self.last_call_time = time.time()

        # Acquire a semaphore permit before calling func
        async with self.semaphore:
            result = await func(*args, **kwargs)

        return result

    async def map(self, func, *args_list):
        coroutines = [self._rate_limited_call(func, *args) for args in zip(*args_list)]

        # Use tqdm progress bar with coroutines
        results = []
        for coroutine in atqdm.as_completed(coroutines):
            result = await coroutine
            results.append(result)

        return results


async def run_pool(api, requests, num_workers, num_requests, period):
    pool = AsyncRateLimitThreadPool(num_workers, num_requests, period)
    writer_task = asyncio.create_task(api.write_responses_to_file())

    results = await pool.map(api.query, requests)
    await api.response_queue.join()  # Ensure all responses are written
    writer_task.cancel()

    return results


def run_openai_inference(
    requests: Union[
        List[Dict], Any
    ],  # can List[Dict] or list of any object with __dict__ attribute
    output_path: str,  # path to save responses
    openai_model: str,  # openai model name
    openai_key: Union[str, List[str]],
    openai_api_base: Optional[Union[str, List[str]]] = None,
    openai_proxy: Optional[Union[str, List[str]]] = None,
    openai_timeout: Optional[float] = 30.0,
    openai_max_retries: Optional[int] = 5,
    generation_config: Optional[Dict] = None,
    num_workers: Optional[int] = 8,
    request_limit: Optional[int] = 100,
    request_limit_period: Optional[int] = 60,
    max_error_count: Optional[int] = 100,
    trial_run=False,
    dump_individual_rsp=True,
):
    logging.getLogger(__name__).info(
        f"num_requests: {len(requests)}, output_path: {output_path}"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    os.makedirs(output_path, exist_ok=True)

    if dump_individual_rsp:
        os.makedirs(os.path.join(output_path, "responses"), exist_ok=True)

    if os.path.exists(os.path.join(output_path, "all_responses.jsonl")):
        os.remove(os.path.join(output_path, "all_responses.jsonl"))

    client = OpenAIClient(
        output_path=output_path,
        openai_model=openai_model,
        openai_key=openai_key,
        openai_api_base=openai_api_base,
        openai_proxy=openai_proxy,
        openai_timeout=openai_timeout,
        openai_max_retries=openai_max_retries,
        generation_config=generation_config,
        trial_run=trial_run,
        dump_individual_rsp=dump_individual_rsp,
    )

    try:
        asyncio.run(
            run_pool(
                client,
                requests,
                num_workers=num_workers,
                num_requests=request_limit,
                period=request_limit_period,
            )
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupt received! Closing...")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="OpenAI ChatCompletion API executor")

    parser.add_argument("-o", "--output-path", default="./responses")
    parser.add_argument("-i", "--input-path", default="", required=True)
    parser.add_argument("-n", "--num-workers", type=int, default=8)
    parser.add_argument("-q", "--num-requests", type=int, default=4000)
    parser.add_argument("-e", "--period", type=int, default=60)
    parser.add_argument("-t", "--trial-run", action="store_true")
    parser.add_argument("-d", "--dump-response", action="store_true")

    args = parser.parse_args()

    with codecs.open(args.input_path) as f:
        requests = json.load(f)

    logging.getLogger(__name__).warning(f"Loaded {len(requests)} requests")

    generation_config = None

    run_openai_inference(
        requests=requests,
        openai_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
        openai_model="gpt-4-1106-preview",
        openai_proxy=os.environ.get("all_proxy", None),
        output_path=args.output_path,
        generation_config=generation_config,
        num_workers=args.num_workers,
        request_limit=args.num_requests,
        request_limit_period=args.period,
        trial_run=args.trial_run,
        dump_individual_rsp=args.dump_response,
    )
