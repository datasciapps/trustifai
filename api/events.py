from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Any, Sequence
import asyncio
from termcolor import colored


@dataclass
class Event:
    type: str
    data: Dict[str, Any] = field(default_factory=dict)


def verbose(event: Event):
    print(colored(f'{event.type:<30}', "green") + f'{event.data}' + "\n")


handlers = defaultdict(lambda: [verbose])


def emit(*events: Sequence[Event]):
    for event in events:
        for fn in handlers[event.type]:
            fn(event)


async def _map(func, iterable, max_workers):
    semaphore, results = asyncio.Semaphore(max_workers), []
    
    async def task(item):
        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, func, item)
        
    async with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [task(item) for item in iterable]
        results = await asyncio.gather(*tasks)
    return results