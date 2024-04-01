"""
Adapted from: https://github.com/TranNhiem/Multimodal_Integrated_App/blob/main/Language/Translate_modules/translate_dataset_NLLB.py
Use this code to process https://huggingface.co/datasets/yahma/alpaca-cleaned
before translations using NLLB models.
The translated datasets and trained LoRAs are available on huggingface
at https://huggingface.co/collections/iamshnoo/alpaca-2-64fe0c729a62bb2791f86745
"""

import re
import string


def matches_regex(regex, text):
    """
    Check if the text matches the given regex pattern.

    Args:
        regex (str): Regular expression pattern.
        text (str): Input text.

    Returns:
        bool: True if the text matches the pattern, False otherwise.
    """
    return bool(re.compile(regex).search(text))


def contains_code(text):
    """
    Check if the text contains code snippets.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text contains code, False otherwise.
    """
    code_blacklist = ["&&", "||", "<html>", ";\n", "SELECT"]
    return (
        any(code_keyword in text for code_keyword in code_blacklist)
        or matches_regex(r"\w+\(\w*\) \{", text)
        or matches_regex(r"def \w+\(", text)
        or matches_regex(r"\[A-z]+\.[A-z]+", text)
        or matches_regex(r": [\w\.#]{1,12};", text)
        or matches_regex(r"<\/\w+>", text)
    )


def contains_words(text):
    """
    Check if the text contains words.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text contains words, False otherwise.
    """
    return matches_regex(r"[A-z]{3,}", text)


def is_translatable(text):
    """
    Check if the given text is translatable.

    Args:
        text (str): Input text.

    Returns:
        bool: True if the text is translatable, False otherwise.
    """
    if text == "":
        return False
    return (contains_code(text) is False) and contains_words(text)
