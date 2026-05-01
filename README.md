# finnhub-mcp

FastMCP server exposing Finnhub API endpoints as MCP tools.

## Setup

```bash
# Install dependencies
pip install -e .

# Set your API key
cp .env.example .env
# Edit .env and add your Finnhub API key
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `FINNHUB_API_KEY` | (required) | Your Finnhub API key |
| `FINNHUB_TIER` | `free` | `free` = ~29 free-tier tools only. `premium` = all ~115 tools. |

Tier is checked at startup. Changing it requires a server restart.

## Running

```bash
# Via fastmcp CLI
fastmcp run server.py

# Or directly
python server.py
```

## Architecture

- **server.py** -- single-file MCP server with all tools
- Uses `httpx.AsyncClient` for async HTTP calls to `https://finnhub.io/api/v1`
- API key injected automatically via `token` query param on every request
- All tools return raw Finnhub JSON responses (dict or list)
- `from`/`to` date params use `from_date`/`to_date` in Python signatures, mapped to `from`/`to` in API requests
- Premium/free classification sourced from `https://finnhub.io/static/swagger.json`

## Adding a new endpoint

```python
@finnhub_tool("/api/path")
async def new_endpoint(symbol: str, optional_param: Optional[str] = None) -> dict | list:
    """Docstring describing what data is returned."""
    return await _get("/api/path", {"symbol": symbol, "param": optional_param})
```

Add the API path to `PREMIUM_ENDPOINTS` if it requires a paid subscription.

## Dependencies

- `fastmcp` -- MCP server framework
- `httpx` -- async HTTP client
- `python-dotenv` -- env file loading
