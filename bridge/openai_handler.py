"""This module handles the communication with the OpenAI API."""
import asyncio
import functools
import copy

import openai
import openai.error

from bridge.config import Config
from bridge.logger import Logger

config = Config()
logger = Logger.get_logger(config.app.name)


openai.api_key = config.openai.api_key
openai.organization = config.openai.organization


async def analyze_message_and_generate_suggestions(text: str) -> str:
    """analyze the message text and seek for suggestions."""

    loop = asyncio.get_event_loop()
    try:
        create_completion = functools.partial(
            openai.Completion.create,
            model="gpt-3.5-turbo-0125",
            prompt=(
                f"Given the message: '{text}', suggest related actions and correlated articles with links:\n"
                f"Related Actions:\n- ACTION1\n- ACTION2\n- ACTION3\n"
                f"Correlated Articles:\n1. ARTICLE1_TITLE - ARTICLE1_LINK\n"
                f"2. ARTICLE2_TITLE - ARTICLE2_LINK\n"
                f"3. ARTICLE3_TITLE - ARTICLE3_LINK\n"
            ),
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        response = await loop.run_in_executor(None, create_completion)

        suggestion = response.choices[0].text.strip() # type: ignore # pylint: disable=no-member
        return suggestion
    except openai.error.InvalidRequestError as ex:
        logger.error("Invalid request error: %s", {ex})
        return "Error generating suggestion: Invalid request."
    except openai.error.RateLimitError as ex:
        logger.error("Rate limit error: %s", {ex})
        return "Error generating suggestion: Rate limit exceeded."
    except openai.error.APIError as ex:
        logger.error("API error: %s", {ex})
        return "Error generating suggestion: API error."
    except Exception as ex:  # pylint: disable=broad-except
        logger.error("Error generating suggestion: %s", {ex})
        return "Error generating suggestion."


async def analyze_message_sentiment(text: str) -> str:
    """analyze the message text and seek for suggestions."""
    loop = asyncio.get_event_loop()
    try:
        prompt = copy.deepcopy(config.openai.sentiment_analysis_prompt)

        if prompt is not None:
            prompt.append({"role":"user","content":text})

        logger.debug("openai_sentiment_analysis_prompt %s", prompt)

        create_completion = functools.partial(
            openai.ChatCompletion.create,
            model=config.openai.model,
            temperature=config.openai.temperature,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=(prompt)
        )

        response = await loop.run_in_executor(None, create_completion)

        suggestion = response.choices[0].message.content # type: ignore # pylint: disable=no-member
        logger.debug("openai_sentiment_analysis_prompt result %s", suggestion)
        return suggestion
    except openai.error.InvalidRequestError as ex:
        logger.error("Invalid request error: %s", {ex})
        return "Error generating suggestion: Invalid request."
    except openai.error.RateLimitError as ex:
        logger.error("Rate limit error: %s", {ex})
        return "Error generating suggestion: Rate limit exceeded."
    except openai.error.APIError as ex:
        logger.error("API error: %s", {ex})
        return "Error generating suggestion: API error."
    except Exception as ex:  # pylint: disable=broad-except
        logger.error("Error generating suggestion: %s", {ex})
        return "Error generating suggestion."
