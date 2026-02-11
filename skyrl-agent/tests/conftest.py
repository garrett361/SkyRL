"""Pytest configuration for skyrl_agent tests.

Sets required environment variables that some tools check at class-body import time
(e.g. SearchEngine requires GOOGLE_SEARCH_KEY).
"""

import os

os.environ.setdefault("GOOGLE_SEARCH_KEY", "test-dummy-key")
os.environ.setdefault("GOOGLE_SEARCH_CX", "test-dummy-cx")
