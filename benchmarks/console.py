"""Simple console color helpers for benchmarks.

Provide small helpers to colorize messages consistently across benchmarks.
Error -> red, Success -> green, Message/info -> grey.
"""
from typing import Any

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
GREY = "\033[90m"


def red(text: Any) -> str:
    return f"{RED}{text}{RESET}"


def green(text: Any) -> str:
    return f"{GREEN}{text}{RESET}"


def grey(text: Any) -> str:
    return f"{GREY}{text}{RESET}"
