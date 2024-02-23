from text_generation import AsyncClient
from tqdm import tqdm
from typing import Optional, List, Dict, Union
from random import choice

import asyncio
import json
import os
import argparse
import logging
import time
import codecs
from typing import Optional, List, Any
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm


class GlobalClientPool:
    def __init__(self):
        self.clients = []
        self.locks = {}
        self.num_requests = {}
        self.pool_lock = asyncio.Lock()  # Add a lock for thread-safe access

    def show_num_requests(self):
        ret = {c.base_url: self.num_requests[c] for c in self.num_requests}
        return ret

    def add_client(self, client_credentials, max_concurrency):
        client = AsyncClient(**client_credentials)
        self.clients.append(client)
        self.locks[client] = asyncio.Semaphore(max_concurrency)

    async def get_client(self):
        async with self.pool_lock:  # Ensure only one coroutine enters this block at a time
            while True:
                # Select the client with the most remaining concurrency
                client = max(self.clients, key=lambda c: self.locks[c]._value)
                semaphore = self.locks[client]

                if semaphore.locked() and semaphore._value <= 0:
                    await asyncio.sleep(0.1)
                    continue

                await semaphore.acquire()
                self.num_requests[client] = self.num_requests.get(client, 0) + 1
                return client

    async def release_client(self, client):
        self.locks[client].release()


class TextGenerationInferenceClient:
    def __init__(
        self,
        output_path: str,
        client_pool: GlobalClientPool,
        generation_config: Dict = None,
        trial_run=False,
        dump_individual_rsp=True,
    ):
        self.output_path = output_path
        self.trial_run = trial_run
        self.logger = logging.getLogger(__name__)

        self.client_pool = client_pool

        self.response_queue = asyncio.Queue()
        self.dump_individual_rsp = dump_individual_rsp

        if generation_config is None:
            self.generation_config = {
                "do_sample": False,
                "max_new_tokens": 20,
                "return_full_text": False,
                "seed": 42,
                "stop_sequences": ["A", "B", "C", "D"],
                "temperature": None,
                "decoder_input_details": False,
            }
        else:
            self.generation_config = generation_config

        if dump_individual_rsp:
            os.makedirs(os.path.join(self.output_path, "responses"), exist_ok=True)

    async def query(
        self,
        request,
        num_retries=1,
    ):
        if isinstance(request, dict):
            request_dict = request
        else:
            request_dict = request.__dict__

        assert "prompt" in request_dict, "prompt must be provided in request"
        assert "uuid" in request_dict, "uuid must be provided in request"

        if self.dump_individual_rsp:
            save_path = os.path.join(
                self.output_path, "responses", f'{request_dict["uuid"]}.json'
            )

            if os.path.exists(save_path) and not self.trial_run:
                try:
                    with codecs.open(save_path) as f:
                        rsp_content = json.load(f)
                except:
                    raise Exception(f"Failed to load cached response: {save_path}")

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
            client = await self.client_pool.get_client()
            try:
                response = await client.generate(
                    prompt=request_dict["prompt"], **generation_config
                )
                response = response.dict()
            except:
                self.logger.exception("Failed to query API, retrying...")
            finally:
                await self.client_pool.release_client(client)
            if response is not None:
                break

        if response is None:
            raise Exception("Empty response from remote hf inference API")

        if self.dump_individual_rsp:
            with codecs.open(save_path, "w") as f:
                json.dump(
                    {"request": request_dict, "response": response},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        await self.response_queue.put({"request": request_dict, "response": response})

        return response["generated_text"]

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


def run_remote_hf_inference(
    requests: Union[
        List[Dict], Any
    ],  # can List[Dict] or list of any object with __dict__ attribute
    output_path: str,  # path to save responses
    base_url: Union[str, List[str]],
    timeout: Optional[Union[int, List[int]]] = 30,
    max_concurrency: Optional[Union[int, List[int]]] = 128,
    generation_config: Optional[Dict] = None,
    num_workers: Optional[int] = 8,
    request_limit: Optional[int] = 100,
    request_limit_period: Optional[int] = 60,
    model_name: Optional[str] = None,
    trial_run: Optional[bool] = False,
    dump_individual_rsp: Optional[bool] = True,
):
    if model_name:
        logging.getLogger(__name__).info(
            f"num_requests: {len(requests)}, model_name: {model_name}, output_path: {output_path}"
        )
    else:
        logging.getLogger(__name__).info(
            f"num_requests: {len(requests)}, base_url: {base_url}, output_path: {output_path}"
        )

    os.makedirs(output_path, exist_ok=True)

    if dump_individual_rsp:
        os.makedirs(os.path.join(output_path, "responses"), exist_ok=True)

    if os.path.exists(os.path.join(output_path, "all_responses.jsonl")):
        os.remove(os.path.join(output_path, "all_responses.jsonl"))

    pool = GlobalClientPool()

    # add clients here

    assert isinstance(timeout, int) or isinstance(
        timeout, list
    ), "timeout must be int or list"
    assert isinstance(max_concurrency, int) or isinstance(
        max_concurrency, list
    ), "max_concurrency must be int or list"

    if isinstance(base_url, str):
        base_url = [base_url]
    if isinstance(timeout, int):
        timeout = [timeout] * len(base_url)
    if isinstance(max_concurrency, int):
        max_concurrency = [max_concurrency] * len(base_url)

    assert (
        len(base_url) == len(timeout) == len(max_concurrency)
    ), "length of base_url, timeout and max_concurrency must be the same"

    for i in range(len(base_url)):
        pool.add_client(
            {
                "base_url": base_url[i],
                "timeout": timeout[i],
            },
            max_concurrency[i],
        )

    api = TextGenerationInferenceClient(
        output_path=output_path,
        client_pool=pool,
        trial_run=trial_run,
        dump_individual_rsp=dump_individual_rsp,
        generation_config=generation_config,
    )

    try:
        asyncio.run(
            run_pool(
                api,
                requests,
                num_workers=num_workers,
                num_requests=request_limit,
                period=request_limit_period,
            )
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupt received! Closing...")

    logging.getLogger(__name__).info(
        f"Requests processed per client: {pool.show_num_requests()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Huggingface text-generation-inference API executor"
    )

    parser.add_argument("-o", "--output-path", default="./responses")
    parser.add_argument("-i", "--input-path", default="", required=True)
    parser.add_argument("-n", "--num-workers", type=int, default=8)
    parser.add_argument("-q", "--num-requests", type=int, default=4000)
    parser.add_argument("-e", "--period", type=int, default=60)
    parser.add_argument("-t", "--trial-run", action="store_true")
    parser.add_argument("-d", "--dump-response", action="store_true")

    args = parser.parse_args()

    base_url = [
        "http://your-tgi-uri:your-tgi-port",
    ]
    timeout = 30
    max_concurrency = 8

    with codecs.open(args.input_path) as f:
        requests = json.load(f)

    logging.getLogger(__name__).warning(f"Loaded {len(requests)} requests")

    generation_config = None

    run_remote_hf_inference(
        requests=requests,
        base_url=base_url,
        timeout=timeout,
        max_concurrency=max_concurrency,
        output_path=args.output_path,
        generation_config=generation_config,
        num_workers=args.num_workers,
        request_limit=args.num_requests,
        request_limit_period=args.period,
        trial_run=args.trial_run,
        dump_individual_rsp=args.dump_response,
    )
