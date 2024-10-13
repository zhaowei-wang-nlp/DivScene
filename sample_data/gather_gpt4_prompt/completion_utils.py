import re
import time
import random
import openai
import asyncio
from string import punctuation

punctuation = set(punctuation)


# define a retry decorator
def retry_with_exponential_backoff(
        initial_delay: float = 1,
        exponential_base: float = 1.1,
        jitter: bool = True,
        max_retries: int = 40,
):
    """Retry a function with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Initialize variables
            rate_limit_retry_num = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)
                # retry on all exceptions and errors
                except Exception as e:
                    print(f"Try count: {rate_limit_retry_num}, Error: {e}")
                    # Increment retries
                    rate_limit_retry_num += 1

                    # Check if max retries has been reached
                    if rate_limit_retry_num > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    print(f"Failure, sleep {delay} secs")
                    time.sleep(delay)
        return wrapper
    return decorator


@retry_with_exponential_backoff
def completions_with_backoff(client, **kwargs):
    return client.completions.create(**kwargs)


@retry_with_exponential_backoff()
def chat_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)
