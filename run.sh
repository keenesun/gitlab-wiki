#!/bin/bash
# Start the DeepWiki backend server.
# Prerequisites: uv (https://docs.astral.sh/uv/)
uv sync
uv run -m api.main
