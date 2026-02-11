import concurrent.futures
import json
import logging
import os
import threading
import time
import traceback
from typing import Any, Optional

import requests

DEFAULT_TIMEOUT = 10
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
TIMEOUT_ERROR_PREFIX = "TIMEOUT: "
RETRYABLE_STATUS_CODES = {502, 503, 504}
HTTP_TIMEOUT_BUFFER = 10

logger = logging.getLogger(__name__)


def call_openenv_api(
    openenv_url: str,
    code: str,
    timeout_s: float = 30.0,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """POST to the OpenEnv /step endpoint with retry logic.

    Returns:
        (response_json, None) on success, or (None, error_msg) on failure.
        Timeout errors are prefixed with TIMEOUT_ERROR_PREFIX.
    """
    endpoint = f"{openenv_url.rstrip('/')}/step"
    body = {"action": {"code": code}, "timeout_s": timeout_s}
    http_timeout = timeout_s + HTTP_TIMEOUT_BUFFER

    last_error: Optional[str] = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(endpoint, json=body, timeout=http_timeout)
        except requests.Timeout as exc:
            last_error = f"{TIMEOUT_ERROR_PREFIX}{exc}"
            logger.warning("Attempt %d/%d timed out: %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue
        except requests.ConnectionError as exc:
            last_error = f"ConnectionError: {exc}"
            logger.warning("Attempt %d/%d connection error: %s", attempt + 1, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            last_error = f"HTTP {response.status_code}: {response.text}"
            logger.warning(
                "Attempt %d/%d: HTTP %d", attempt + 1, MAX_RETRIES, response.status_code
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_RETRY_DELAY * (attempt + 1))
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            return None, f"HTTP {response.status_code}: {response.text}"

        try:
            return response.json(), None
        except json.JSONDecodeError as exc:
            return None, f"JSON decode error: {exc}"

    logger.error("OpenEnv API call failed after %d attempts. Last error: %s", MAX_RETRIES, last_error)
    return None, last_error


def _build_fn_wrapper(generation: str, fn_name: str, stdin_data: Any) -> str:
    """Build the function-call wrapper with stdin embedded as a string literal."""
    embedded_input = json.dumps(str(stdin_data))
    return f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    _raw_input_str = {embedded_input}
    _args = []
    if _raw_input_str.strip():
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                              f"{{_raw_input_str[:200]}}\\n")
            return None, True

    try:
        _target_callable = None
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True

        _fn_result = _target_callable(*_args)
        return _fn_result, False
    except Exception:
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result))
        else:
            print(str(_result))
"""


def _process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    openenv_url: str,
    generation: str,
    timeout: int,
    fn_name: Optional[str] = None,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[int | bool, dict[str, Any]]:
    """Process a single test case against the OpenEnv API."""
    code = generation
    if fn_name:
        code = _build_fn_wrapper(generation, fn_name, stdin_data)

    error_msg = None
    api_response = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_openenv_api(
                    openenv_url, code, timeout_s=float(timeout)
                )
        else:
            api_response, error_msg = call_openenv_api(
                openenv_url, code, timeout_s=float(timeout)
            )
    except Exception as exc:
        error_msg = f"Exception processing case {case_index}: {exc}"
        logger.error(error_msg)
        traceback.print_exc()

    metadata: dict[str, Any] = {
        "case_index": case_index,
        "input": None if stdin_data is None else str(stdin_data),
        "expected_output": str(expected_output),
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "status": "unknown",
        "api_request_error": error_msg,
    }

    if error_msg:
        if error_msg.startswith(TIMEOUT_ERROR_PREFIX):
            metadata["status"] = "timeout"
            return -3, metadata
        metadata["status"] = "api_error"
        return -1, metadata

    if not api_response:
        metadata["status"] = "api_error"
        return -1, metadata

    obs = api_response.get("observation", {})
    stdout = obs.get("stdout", "")
    stderr = obs.get("stderr", "")
    exit_code = obs.get("exit_code", -1)

    metadata["stdout"] = stdout
    metadata["stderr"] = stderr
    metadata["exit_code"] = exit_code

    if exit_code != 0:
        metadata["status"] = "runtime_error"
        return -2, metadata

    if str(stdout).rstrip("\n") == str(expected_output).rstrip("\n"):
        metadata["status"] = "success"
        return True, metadata

    metadata["status"] = "wrong_answer"
    return False, metadata


def check_correctness(
    openenv_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Check correctness of generated code against all test cases.

    Returns:
        (results, metadata_list) where each result is True/False/-1/-2/-3.
    """
    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)

    if num_cases == 0:
        return [], []

    results: list[Any] = [None] * num_cases
    metadata_list: list[Any] = [None] * num_cases

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(32, (os.cpu_count() or 1) * 5)
    ) as executor:
        future_to_index = {
            executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                openenv_url,
                generation,
                timeout,
                fn_name,
                concurrent_semaphore,
            ): i
            for i, stdin_data in enumerate(inputs)
        }

        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata
            except Exception as exc:
                logger.error("Test case %d generated an exception: %s", index, exc)
                traceback.print_exc()
                results[index] = -1
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]),
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

    return results, metadata_list
