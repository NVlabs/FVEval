import os
import pathlib
import time
import random

import pandas as pd
from tqdm import tqdm

from adlrchat.langchain import ADLRChat, LLMGatewayChat
from langchain.schema import HumanMessage, SystemMessage

from fv_eval import utils
from typing import Any, Callable, Collection, Generic, List, Optional, Type, TypeVar

Q = TypeVar("Q", bound=Callable[..., Any])

# define a retry decorator
def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    no_retry_on: Optional[Collection[Type[Exception]]] = None,
)->Callable[[Q], Q]:
    """Retry a function with exponential backoff."""
    def decorator(func: Q)->Q:
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            error = None
    
            # Loop until a successful response or max_retries is hit or an exception is raised
            while num_retries <= max_retries:
                try:
                    return func(*args, **kwargs)
    
                # Raise exceptions for any errors specified
                except Exception as e:
                    if no_retry_on is not None and type(e) in no_retry_on:
                        raise e
    
                    # Sleep for the delay
                    time.sleep(delay)

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Set the error to the last exception
                    error = e

                    # Increment retries
                    num_retries += 1

                    print(f"Retrying {func.__name__} after error: {e} (retry {num_retries} of {max_retries})")

            if error is not None:
                raise error
        return wrapper
    return decorator

def lm_inference_adlrchat(
    model_name: str = "gpt-4",
    temperature: float = 0.0,
    system_prompt: str = "",
    user_prompt: str = "",
    max_tokens: int = 100,
):
    if "gpt" in model_name:
        chat = LLMGatewayChat(
            streaming=True,
            temperature=temperature,
            model=model_name,
            max_tokens=max_tokens,
        )
    else:
        chat = ADLRChat(
            streaming=True,
            temperature=temperature,
            model=model_name,
            max_tokens=max_tokens,
        )
    try:
        lm_response = chat(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return lm_response.content
    except:
        return ""
