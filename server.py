"""FastMCP server wrapping Finnhub API endpoints.

Set FINNHUB_TIER=free (default) to expose only free-tier endpoints,
or FINNHUB_TIER=premium to expose all endpoints.

Run with:
    fastmcp run server.py
    python server.py
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
if not FINNHUB_API_KEY:
    print("ERROR: FINNHUB_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

FINNHUB_TIER = os.environ.get("FINNHUB_TIER", "free").lower()
if FINNHUB_TIER not in ("free", "premium"):
    print(
        f"ERROR: FINNHUB_TIER must be 'free' or 'premium', got '{FINNHUB_TIER}'",
        file=sys.stderr,
    )
    sys.exit(1)

BASE_URL = "https://finnhub.io/api/v1"


@asynccontextmanager
async def _lifespan(app):
    yield
    await _close_client()


mcp = FastMCP("finnhub", lifespan=_lifespan)

# ---------------------------------------------------------------------------
# Premium endpoint classification (from finnhub.io/static/swagger.json)
# ---------------------------------------------------------------------------

PREMIUM_ENDPOINTS: frozenset[str] = frozenset(
    {
        "/airline/price-index",
        "/bank-branch",
        "/bond/price",
        "/bond/profile",
        "/bond/tick",
        "/bond/yield-curve",
        "/ca/isin-change",
        "/ca/symbol-change",
        "/calendar/economic",
        "/crypto/candle",
        "/crypto/profile",
        "/economic",
        "/economic/code",
        "/etf/allocation",
        "/etf/country",
        "/etf/holdings",
        "/etf/profile",
        "/etf/sector",
        "/forex/candle",
        "/forex/rates",
        "/global-filings/download",
        "/global-filings/filter",
        "/global-filings/search",
        "/global-filings/search-in-filing",
        "/index/constituents",
        "/index/historical-constituents",
        "/indicator",
        "/institutional/ownership",
        "/institutional/portfolio",
        "/institutional/profile",
        "/mutual-fund/country",
        "/mutual-fund/eet",
        "/mutual-fund/eet-pai",
        "/mutual-fund/holdings",
        "/mutual-fund/profile",
        "/mutual-fund/sector",
        "/news-sentiment",
        "/press-releases",
        "/scan/pattern",
        "/scan/support-resistance",
        "/scan/technical-indicator",
        "/sector/metrics",
        "/stock/bbo",
        "/stock/bidask",
        "/stock/candle",
        "/stock/congressional-trading",
        "/stock/dividend",
        "/stock/dividend2",
        "/stock/dps-estimate",
        "/stock/earnings-call-live",
        "/stock/earnings-quality-score",
        "/stock/ebit-estimate",
        "/stock/ebitda-estimate",
        "/stock/eps-estimate",
        "/stock/esg",
        "/stock/exchange",
        "/stock/executive",
        "/stock/filings-sentiment",
        "/stock/financials",
        "/stock/fund-ownership",
        "/stock/gross-income-estimate",
        "/stock/historical-employee-count",
        "/stock/historical-esg",
        "/stock/historical-market-cap",
        "/stock/international-filings",
        "/stock/investment-theme",
        "/stock/net-income-estimate",
        "/stock/newsroom",
        "/stock/option-chain",
        "/stock/ownership",
        "/stock/presentation",
        "/stock/pretax-income-estimate",
        "/stock/price-metric",
        "/stock/price-target",
        "/stock/profile",
        "/stock/revenue-breakdown",
        "/stock/revenue-breakdown2",
        "/stock/revenue-estimate",
        "/stock/similarity-index",
        "/stock/social-sentiment",
        "/stock/split",
        "/stock/supply-chain",
        "/stock/tick",
        "/stock/transcripts",
        "/stock/transcripts/list",
        "/stock/upgrade-downgrade",
    }
)


def finnhub_tool(api_path: str) -> Callable:
    """Conditionally register an MCP tool based on the configured tier.

    Free-tier tools are always registered. Premium tools are only registered
    when ``FINNHUB_TIER=premium``.
    """
    is_premium = api_path in PREMIUM_ENDPOINTS

    def decorator(func: Callable) -> Callable:
        if is_premium and FINNHUB_TIER != "premium":
            return func
        return mcp.tool()(func)

    return decorator

# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    """Return (and lazily create) a shared async HTTP client."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=30.0,
            headers={"X-Finnhub-Token": FINNHUB_API_KEY},
        )
    return _client


async def _close_client() -> None:
    """Close the shared HTTP client if it exists."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


async def _get(path: str, params: dict[str, Any] | None = None) -> dict | list:
    """Make a GET request to the Finnhub API.

    Automatically strips None-valued params.

    Args:
        path: API endpoint path (e.g. ``/quote``).
        params: Query parameters. ``None`` values are removed before sending.

    Returns:
        Parsed JSON response from Finnhub.
    """
    client = await _get_client()
    clean: dict[str, Any] = {}
    if params:
        for k, v in params.items():
            if v is not None:
                clean[k] = v
    resp = await client.get(path, params=clean)
    resp.raise_for_status()
    return resp.json()


async def _post(path: str, body: dict[str, Any] | None = None) -> dict | list:
    """Make a POST request to the Finnhub API.

    Args:
        path: API endpoint path.
        body: JSON request body. ``None`` values are removed before sending.

    Returns:
        Parsed JSON response from Finnhub.
    """
    client = await _get_client()
    clean: dict[str, Any] = {}
    if body:
        for k, v in body.items():
            if v is not None:
                clean[k] = v
    resp = await client.post(path, json=clean)
    resp.raise_for_status()
    return resp.json()


# ===================================================================
# STOCK DATA
# ===================================================================


@finnhub_tool("/stock/profile")
async def company_profile(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
    cusip: Optional[str] = None,
) -> dict | list:
    """Get general information of a company.

    Provide at least one identifier. Data includes company name, ticker,
    exchange, industry, IPO date, market cap, and more.

    Args:
        symbol: Stock symbol (e.g. ``AAPL``).
        isin: ISIN identifier.
        cusip: CUSIP identifier.
    """
    return await _get("/stock/profile", {"symbol": symbol, "isin": isin, "cusip": cusip})


@finnhub_tool("/stock/profile2")
async def company_profile2(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
    cusip: Optional[str] = None,
) -> dict | list:
    """Get general information of a company (version 2, free tier).

    Similar to ``company_profile`` but available on the free tier.

    Args:
        symbol: Stock symbol (e.g. ``AAPL``).
        isin: ISIN identifier.
        cusip: CUSIP identifier.
    """
    return await _get("/stock/profile2", {"symbol": symbol, "isin": isin, "cusip": cusip})


@finnhub_tool("/scan/technical-indicator")
async def aggregate_indicator(symbol: str, resolution: str) -> dict | list:
    """Get aggregate signal of multiple technical indicators.

    Args:
        symbol: Stock symbol (e.g. ``AAPL``).
        resolution: Candle resolution (``1``, ``5``, ``15``, ``30``, ``60``, ``D``, ``W``, ``M``).
    """
    return await _get("/scan/technical-indicator", {"symbol": symbol, "resolution": resolution})


@finnhub_tool("/stock/executive")
async def company_executive(symbol: str) -> dict | list:
    """List company executives and board members.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/executive", {"symbol": symbol})


@finnhub_tool("/stock/dividend")
async def stock_dividends(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get dividends data for a stock.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/stock/dividend", {"symbol": symbol, "from": from_date, "to": to_date})


@finnhub_tool("/stock/dividend2")
async def stock_basic_dividends(symbol: str) -> dict | list:
    """Get basic dividend data for a stock (version 2).

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/dividend2", {"symbol": symbol})


@finnhub_tool("/stock/symbol")
async def stock_symbols(
    exchange: str,
    mic: Optional[str] = None,
    security_type: Optional[str] = None,
    currency: Optional[str] = None,
) -> dict | list:
    """List supported stocks for an exchange.

    Args:
        exchange: Exchange code (e.g. ``US``).
        mic: Market identifier code.
        security_type: Security type filter.
        currency: Currency filter.
    """
    return await _get(
        "/stock/symbol",
        {"exchange": exchange, "mic": mic, "securityType": security_type, "currency": currency},
    )


@finnhub_tool("/stock/recommendation")
async def recommendation_trends(symbol: str) -> dict | list:
    """Get latest analyst recommendation trends (buy, hold, sell, etc.).

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/recommendation", {"symbol": symbol})


@finnhub_tool("/stock/price-target")
async def price_target(symbol: str) -> dict | list:
    """Get latest price target consensus from analysts.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/price-target", {"symbol": symbol})


@finnhub_tool("/stock/upgrade-downgrade")
async def upgrade_downgrade(
    symbol: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get latest stock upgrade/downgrade from analysts.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/stock/upgrade-downgrade", {"symbol": symbol, "from": from_date, "to": to_date})


@finnhub_tool("/stock/option-chain")
async def option_chain(
    symbol: str,
    date: Optional[str] = None,
    option_type: Optional[str] = None,
    strike: Optional[float] = None,
) -> dict | list:
    """Get option chain data for a symbol.

    Args:
        symbol: Stock symbol.
        date: Expiration date (``YYYY-MM-DD``).
        option_type: Option type: ``call`` or ``put``.
        strike: Strike price to filter.
    """
    return await _get(
        "/stock/option-chain",
        {"symbol": symbol, "date": date, "optionType": option_type, "strike": strike},
    )


@finnhub_tool("/stock/peers")
async def company_peers(symbol: str) -> dict | list:
    """Get a list of peers/comparable companies for a symbol.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/peers", {"symbol": symbol})


@finnhub_tool("/stock/metric")
async def company_basic_financials(symbol: str, metric: str = "all") -> dict | list:
    """Get basic financial metrics, margin, profitability, and valuation data.

    Args:
        symbol: Stock symbol.
        metric: Metric type (``all``, ``price``, ``valuation``, ``margin``).
    """
    return await _get("/stock/metric", {"symbol": symbol, "metric": metric})


@finnhub_tool("/stock/financials")
async def financials(
    symbol: str,
    statement: str,
    freq: str,
    preliminary: Optional[bool] = None,
) -> dict | list:
    """Get standardized financial statements.

    Args:
        symbol: Stock symbol.
        statement: Statement type (``bs`` for balance sheet, ``ic`` for income, ``cf`` for cash flow).
        freq: Frequency (``annual``, ``quarterly``, ``ttm``, ``ytd``).
        preliminary: Include preliminary data.
    """
    return await _get(
        "/stock/financials",
        {"symbol": symbol, "statement": statement, "freq": freq, "preliminary": preliminary},
    )


@finnhub_tool("/stock/financials-reported")
async def financials_reported(
    symbol: Optional[str] = None,
    cik: Optional[str] = None,
    freq: Optional[str] = None,
    access_number: Optional[str] = None,
) -> dict | list:
    """Get as-reported financial statements from SEC filings.

    Args:
        symbol: Stock symbol.
        cik: CIK number.
        freq: Frequency (``annual``, ``quarterly``).
        access_number: SEC access number for a specific filing.
    """
    return await _get(
        "/stock/financials-reported",
        {"symbol": symbol, "cik": cik, "freq": freq, "accessNumber": access_number},
    )


@finnhub_tool("/stock/fund-ownership")
async def fund_ownership(symbol: str, limit: Optional[int] = None) -> dict | list:
    """Get a list of mutual funds holding a particular stock.

    Args:
        symbol: Stock symbol.
        limit: Maximum number of results.
    """
    return await _get("/stock/fund-ownership", {"symbol": symbol, "limit": limit})


@finnhub_tool("/stock/earnings")
async def company_earnings(symbol: str, limit: Optional[int] = None) -> dict | list:
    """Get historical earnings surprises for a company.

    Args:
        symbol: Stock symbol.
        limit: Maximum number of quarterly results.
    """
    return await _get("/stock/earnings", {"symbol": symbol, "limit": limit})


@finnhub_tool("/quote")
async def quote(symbol: str) -> dict | list:
    """Get real-time quote data for a stock.

    Returns current price, change, percent change, high, low, open, and
    previous close price, plus timestamp.

    Args:
        symbol: Stock symbol (e.g. ``AAPL``).
    """
    return await _get("/quote", {"symbol": symbol})


@finnhub_tool("/stock/candle")
async def stock_candles(
    symbol: str,
    resolution: str,
    from_date: int,
    to_date: int,
) -> dict | list:
    """Get candlestick data (OHLCV) for stocks.

    Args:
        symbol: Stock symbol.
        resolution: Candle resolution (``1``, ``5``, ``15``, ``30``, ``60``, ``D``, ``W``, ``M``).
        from_date: UNIX timestamp for the start of the range.
        to_date: UNIX timestamp for the end of the range.
    """
    return await _get(
        "/stock/candle",
        {"symbol": symbol, "resolution": resolution, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/tick")
async def stock_tick(
    symbol: str,
    date: str,
    limit: Optional[int] = None,
    skip: Optional[int] = None,
    response_format: Optional[str] = None,
) -> dict | list:
    """Get historical tick data for US stocks.

    Args:
        symbol: Stock symbol.
        date: Date (``YYYY-MM-DD``).
        limit: Maximum number of ticks.
        skip: Number of ticks to skip.
        response_format: Response format (``json`` or ``csv``).
    """
    return await _get(
        "/stock/tick",
        {"symbol": symbol, "date": date, "limit": limit, "skip": skip, "format": response_format},
    )


@finnhub_tool("/stock/bbo")
async def stock_nbbo(
    symbol: str,
    date: str,
    limit: Optional[int] = None,
    skip: Optional[int] = None,
    response_format: Optional[str] = None,
) -> dict | list:
    """Get historical best bid/offer (NBBO) data for US stocks.

    Args:
        symbol: Stock symbol.
        date: Date (``YYYY-MM-DD``).
        limit: Maximum number of records.
        skip: Number of records to skip.
        response_format: Response format (``json`` or ``csv``).
    """
    return await _get(
        "/stock/bbo",
        {"symbol": symbol, "date": date, "limit": limit, "skip": skip, "format": response_format},
    )


@finnhub_tool("/stock/split")
async def stock_splits(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get stock split history.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/stock/split", {"symbol": symbol, "from": from_date, "to": to_date})


@finnhub_tool("/stock/ownership")
async def ownership(symbol: str, limit: Optional[int] = None) -> dict | list:
    """Get institutional ownership data for a stock.

    Args:
        symbol: Stock symbol.
        limit: Maximum number of results.
    """
    return await _get("/stock/ownership", {"symbol": symbol, "limit": limit})


@finnhub_tool("/stock/insider-transactions")
async def stock_insider_transactions(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get insider transactions (SEC Form 4) for a company.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/insider-transactions",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/revenue-breakdown")
async def stock_revenue_breakdown(
    symbol: Optional[str] = None,
    cik: Optional[str] = None,
) -> dict | list:
    """Get revenue breakdown by product/segment/geography.

    Args:
        symbol: Stock symbol.
        cik: CIK number.
    """
    return await _get("/stock/revenue-breakdown", {"symbol": symbol, "cik": cik})


@finnhub_tool("/stock/social-sentiment")
async def stock_social_sentiment(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get social media sentiment data for a stock (Reddit, Twitter).

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/social-sentiment",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/investment-theme")
async def stock_investment_theme(theme: str) -> dict | list:
    """Get stocks belonging to an investment theme.

    Args:
        theme: Investment theme (e.g. ``financialExchange``).
    """
    return await _get("/stock/investment-theme", {"theme": theme})


@finnhub_tool("/stock/supply-chain")
async def stock_supply_chain(symbol: str) -> dict | list:
    """Get supply chain relationships for a company.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/supply-chain", {"symbol": symbol})


@finnhub_tool("/stock/historical-market-cap")
async def historical_market_cap(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get historical market capitalization for a company.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/historical-market-cap",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/uspto-patent")
async def stock_uspto_patent(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get USPTO patent data for a company.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/uspto-patent",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/visa-application")
async def stock_visa_application(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get H1-B visa application data for a company.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/visa-application",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/insider-sentiment")
async def stock_insider_sentiment(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get insider sentiment data based on SEC filings.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/insider-sentiment",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/lobbying")
async def stock_lobbying(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get lobbying activities data for a company.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/lobbying",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/usa-spending")
async def stock_usa_spending(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get government spending data related to a company (USAspending.gov).

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/usa-spending",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/congressional-trading")
async def congressional_trading(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get congressional trading disclosures for a stock.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/congressional-trading",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/market-status")
async def market_status(exchange: str) -> dict | list:
    """Check if a market/exchange is currently open or closed.

    Args:
        exchange: Exchange code (e.g. ``US``).
    """
    return await _get("/stock/market-status", {"exchange": exchange})


@finnhub_tool("/stock/market-holiday")
async def market_holiday(exchange: str) -> dict | list:
    """Get market holiday calendar for an exchange.

    Args:
        exchange: Exchange code (e.g. ``US``).
    """
    return await _get("/stock/market-holiday", {"exchange": exchange})


@finnhub_tool("/stock/historical-employee-count")
async def historical_employee_count(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get historical employee count for a company.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/historical-employee-count",
        {"symbol": symbol, "from": from_date, "to": to_date},
    )


@finnhub_tool("/stock/earnings-call-live")
async def earnings_call_live(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    symbol: Optional[str] = None,
) -> dict | list:
    """Get live earnings call events and audio links.

    Args:
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
        symbol: Stock symbol to filter.
    """
    return await _get(
        "/stock/earnings-call-live",
        {"from": from_date, "to": to_date, "symbol": symbol},
    )


@finnhub_tool("/stock/presentation")
async def stock_presentation(symbol: str) -> dict | list:
    """Get investor presentations and slide decks for a company.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/presentation", {"symbol": symbol})


@finnhub_tool("/stock/revenue-breakdown2")
async def stock_revenue_breakdown2(symbol: str) -> dict | list:
    """Get detailed revenue breakdown (version 2).

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/revenue-breakdown2", {"symbol": symbol})


@finnhub_tool("/stock/newsroom")
async def newsroom(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get company newsroom articles.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/stock/newsroom", {"symbol": symbol, "from": from_date, "to": to_date})


@finnhub_tool("/stock/bidask")
async def last_bid_ask(symbol: str) -> dict | list:
    """Get last bid/ask data for US stocks.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/bidask", {"symbol": symbol})


@finnhub_tool("/stock/price-metric")
async def price_metrics(
    symbol: str,
    date: Optional[str] = None,
) -> dict | list:
    """Get price performance metrics (52-week high/low, beta, etc.).

    Args:
        symbol: Stock symbol.
        date: Date for point-in-time data (``YYYY-MM-DD``).
    """
    return await _get("/stock/price-metric", {"symbol": symbol, "date": date})


# ===================================================================
# ESTIMATES
# ===================================================================


@finnhub_tool("/stock/revenue-estimate")
async def company_revenue_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst revenue estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/revenue-estimate", {"symbol": symbol, "freq": freq})


@finnhub_tool("/stock/ebitda-estimate")
async def company_ebitda_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst EBITDA estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/ebitda-estimate", {"symbol": symbol, "freq": freq})


@finnhub_tool("/stock/ebit-estimate")
async def company_ebit_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst EBIT estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/ebit-estimate", {"symbol": symbol, "freq": freq})


@finnhub_tool("/stock/eps-estimate")
async def company_eps_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst EPS (earnings per share) estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/eps-estimate", {"symbol": symbol, "freq": freq})


@finnhub_tool("/stock/net-income-estimate")
async def company_net_income_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst net income estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/net-income-estimate", {"symbol": symbol, "freq": freq})


@finnhub_tool("/stock/pretax-income-estimate")
async def company_pretax_income_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst pre-tax income estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/pretax-income-estimate", {"symbol": symbol, "freq": freq})


@finnhub_tool("/stock/gross-income-estimate")
async def company_gross_income_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst gross income estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/gross-income-estimate", {"symbol": symbol, "freq": freq})


@finnhub_tool("/stock/dps-estimate")
async def company_dps_estimates(
    symbol: str,
    freq: Optional[str] = None,
) -> dict | list:
    """Get analyst dividends per share (DPS) estimates for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/dps-estimate", {"symbol": symbol, "freq": freq})


# ===================================================================
# ESG
# ===================================================================


@finnhub_tool("/stock/esg")
async def company_esg_score(symbol: str) -> dict | list:
    """Get ESG (Environmental, Social, Governance) score for a company.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/esg", {"symbol": symbol})


@finnhub_tool("/stock/historical-esg")
async def company_historical_esg_score(symbol: str) -> dict | list:
    """Get historical ESG scores for a company.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/historical-esg", {"symbol": symbol})


@finnhub_tool("/stock/earnings-quality-score")
async def company_earnings_quality_score(symbol: str, freq: str) -> dict | list:
    """Get earnings quality score for a company.

    Args:
        symbol: Stock symbol.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get("/stock/earnings-quality-score", {"symbol": symbol, "freq": freq})


# ===================================================================
# EXCHANGE & SEARCH
# ===================================================================


@finnhub_tool("/stock/exchange")
async def exchange() -> dict | list:
    """List supported stock exchanges."""
    return await _get("/stock/exchange")


@finnhub_tool("/search")
async def symbol_lookup(query: str) -> dict | list:
    """Search for symbols by name or ticker.

    Args:
        query: Search query string (company name or partial ticker).
    """
    return await _get("/search", {"q": query})


# ===================================================================
# FILINGS & TRANSCRIPTS
# ===================================================================


@finnhub_tool("/stock/filings")
async def filings(
    symbol: Optional[str] = None,
    cik: Optional[str] = None,
    access_number: Optional[str] = None,
    form: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get SEC filings for a company.

    Args:
        symbol: Stock symbol.
        cik: CIK number.
        access_number: SEC access number.
        form: Filing form type (e.g. ``10-K``, ``10-Q``).
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/stock/filings",
        {
            "symbol": symbol,
            "cik": cik,
            "accessNumber": access_number,
            "form": form,
            "from": from_date,
            "to": to_date,
        },
    )


@finnhub_tool("/stock/transcripts")
async def transcripts(transcript_id: str) -> dict | list:
    """Get full earnings call transcript by ID.

    Args:
        transcript_id: Transcript ID (obtain from ``transcripts_list``).
    """
    return await _get("/stock/transcripts", {"id": transcript_id})


@finnhub_tool("/stock/transcripts/list")
async def transcripts_list(symbol: str) -> dict | list:
    """List available earnings call transcripts for a company.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/stock/transcripts/list", {"symbol": symbol})


@finnhub_tool("/stock/international-filings")
async def international_filings(
    symbol: Optional[str] = None,
    country: Optional[str] = None,
) -> dict | list:
    """Get international filings for a company.

    Args:
        symbol: Stock symbol.
        country: Country code (2-letter ISO).
    """
    return await _get("/stock/international-filings", {"symbol": symbol, "country": country})


@finnhub_tool("/stock/filings-sentiment")
async def sec_sentiment_analysis(access_number: str) -> dict | list:
    """Get SEC filing sentiment analysis.

    Args:
        access_number: SEC filing access number.
    """
    return await _get("/stock/filings-sentiment", {"accessNumber": access_number})


@finnhub_tool("/stock/similarity-index")
async def sec_similarity_index(
    symbol: Optional[str] = None,
    cik: Optional[str] = None,
    freq: Optional[str] = None,
) -> dict | list:
    """Get SEC filing similarity index to detect material changes.

    Args:
        symbol: Stock symbol.
        cik: CIK number.
        freq: Frequency (``annual``, ``quarterly``).
    """
    return await _get(
        "/stock/similarity-index",
        {"symbol": symbol, "cik": cik, "freq": freq},
    )


@finnhub_tool("/press-releases")
async def press_releases(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get press releases for a company.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/press-releases", {"symbol": symbol, "from": from_date, "to": to_date})


# ===================================================================
# GLOBAL FILINGS
# ===================================================================


@finnhub_tool("/global-filings/search")
async def global_filings_search(
    query: str,
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
    country: Optional[str] = None,
    form_type: Optional[str] = None,
    source: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page: Optional[int] = None,
    sort: Optional[str] = None,
) -> dict | list:
    """Search global filings, transcripts, and press releases.

    Args:
        query: Search query text.
        symbol: Stock symbol filter.
        isin: ISIN filter.
        country: Country code filter (2-letter ISO).
        form_type: Form type filter (e.g. ``10-K``, ``10-Q``).
        source: Document source filter.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
        page: Page number for pagination.
        sort: Sort order.
    """
    return await _post(
        "/global-filings/search",
        {
            "query": query,
            "symbol": symbol,
            "isin": isin,
            "country": country,
            "formType": form_type,
            "source": source,
            "from": from_date,
            "to": to_date,
            "page": page,
            "sort": sort,
        },
    )


@finnhub_tool("/global-filings/filter")
async def global_filings_filter(
    field: str,
    source: Optional[str] = None,
) -> dict | list:
    """Get available filter values for global filings search.

    Args:
        field: Filter field (``countries``, ``exchanges``, ``exhibits``, ``forms``).
        source: Get available forms for a specific source.
    """
    return await _get("/global-filings/filter", {"field": field, "source": source})


@finnhub_tool("/global-filings/download")
async def global_filings_download(document_id: str) -> dict | list:
    """Download a global filing document by ID.

    Args:
        document_id: Document ID (different from filing ID; one filing can contain multiple documents).
    """
    return await _get("/global-filings/download", {"documentId": document_id})


@finnhub_tool("/global-filings/search-in-filing")
async def search_in_filing(
    filing_id: str,
    query: str,
) -> dict | list:
    """Search within a specific filing document for excerpts matching a query.

    Args:
        filing_id: Filing ID to search within.
        query: Search query text.
    """
    return await _post(
        "/global-filings/search-in-filing",
        {"filingId": filing_id, "query": query},
    )


# ===================================================================
# CRYPTO
# ===================================================================


@finnhub_tool("/crypto/exchange")
async def crypto_exchanges() -> dict | list:
    """List supported crypto exchanges."""
    return await _get("/crypto/exchange")


@finnhub_tool("/crypto/symbol")
async def crypto_symbols(exchange: str) -> dict | list:
    """List supported crypto symbols for an exchange.

    Args:
        exchange: Crypto exchange name (e.g. ``binance``).
    """
    return await _get("/crypto/symbol", {"exchange": exchange})


@finnhub_tool("/crypto/candle")
async def crypto_candles(
    symbol: str,
    resolution: str,
    from_date: int,
    to_date: int,
) -> dict | list:
    """Get cryptocurrency candlestick data (OHLCV).

    Args:
        symbol: Crypto symbol (e.g. ``BINANCE:BTCUSDT``).
        resolution: Candle resolution (``1``, ``5``, ``15``, ``30``, ``60``, ``D``, ``W``, ``M``).
        from_date: UNIX timestamp for the start.
        to_date: UNIX timestamp for the end.
    """
    return await _get(
        "/crypto/candle",
        {"symbol": symbol, "resolution": resolution, "from": from_date, "to": to_date},
    )


@finnhub_tool("/crypto/profile")
async def crypto_profile(symbol: str) -> dict | list:
    """Get profile/metadata for a cryptocurrency.

    Args:
        symbol: Crypto symbol (e.g. ``BTC``).
    """
    return await _get("/crypto/profile", {"symbol": symbol})


# ===================================================================
# FOREX
# ===================================================================


@finnhub_tool("/forex/exchange")
async def forex_exchanges() -> dict | list:
    """List supported forex exchanges."""
    return await _get("/forex/exchange")


@finnhub_tool("/forex/rates")
async def forex_rates(
    base: Optional[str] = None,
    date: Optional[str] = None,
) -> dict | list:
    """Get forex exchange rates.

    Args:
        base: Base currency (e.g. ``USD``).
        date: Date for historical rates (``YYYY-MM-DD``).
    """
    return await _get("/forex/rates", {"base": base, "date": date})


@finnhub_tool("/forex/symbol")
async def forex_symbols(exchange: str) -> dict | list:
    """List supported forex symbols for an exchange.

    Args:
        exchange: Forex exchange name (e.g. ``oanda``).
    """
    return await _get("/forex/symbol", {"exchange": exchange})


@finnhub_tool("/forex/candle")
async def forex_candles(
    symbol: str,
    resolution: str,
    from_date: int,
    to_date: int,
) -> dict | list:
    """Get forex candlestick data (OHLCV).

    Args:
        symbol: Forex symbol (e.g. ``OANDA:EUR_USD``).
        resolution: Candle resolution (``1``, ``5``, ``15``, ``30``, ``60``, ``D``, ``W``, ``M``).
        from_date: UNIX timestamp for the start.
        to_date: UNIX timestamp for the end.
    """
    return await _get(
        "/forex/candle",
        {"symbol": symbol, "resolution": resolution, "from": from_date, "to": to_date},
    )


# ===================================================================
# TECHNICAL ANALYSIS
# ===================================================================


@finnhub_tool("/scan/pattern")
async def pattern_recognition(symbol: str, resolution: str) -> dict | list:
    """Run pattern recognition on a stock chart.

    Args:
        symbol: Stock symbol.
        resolution: Candle resolution (``1``, ``5``, ``15``, ``30``, ``60``, ``D``, ``W``, ``M``).
    """
    return await _get("/scan/pattern", {"symbol": symbol, "resolution": resolution})


@finnhub_tool("/scan/support-resistance")
async def support_resistance(symbol: str, resolution: str) -> dict | list:
    """Get support and resistance levels for a stock.

    Args:
        symbol: Stock symbol.
        resolution: Candle resolution (``1``, ``5``, ``15``, ``30``, ``60``, ``D``, ``W``, ``M``).
    """
    return await _get("/scan/support-resistance", {"symbol": symbol, "resolution": resolution})


@finnhub_tool("/indicator")
async def technical_indicator(
    symbol: str,
    resolution: str,
    from_date: int,
    to_date: int,
    indicator: str,
    indicator_fields: Optional[dict[str, Any]] = None,
) -> dict | list:
    """Calculate a technical indicator for a symbol.

    Supports indicators like SMA, EMA, RSI, MACD, etc.

    Args:
        symbol: Stock symbol.
        resolution: Candle resolution (``1``, ``5``, ``15``, ``30``, ``60``, ``D``, ``W``, ``M``).
        from_date: UNIX timestamp for the start.
        to_date: UNIX timestamp for the end.
        indicator: Indicator name (e.g. ``sma``, ``ema``, ``rsi``, ``macd``).
        indicator_fields: Additional indicator-specific parameters
            (e.g. ``{"timeperiod": 14}`` for RSI).
    """
    params: dict[str, Any] = {
        "symbol": symbol,
        "resolution": resolution,
        "from": from_date,
        "to": to_date,
        "indicator": indicator,
    }
    if indicator_fields:
        params.update(indicator_fields)
    return await _get("/indicator", params)


# ===================================================================
# NEWS
# ===================================================================


@finnhub_tool("/news")
async def general_news(
    category: str,
    min_id: Optional[int] = None,
) -> dict | list:
    """Get latest market news.

    Args:
        category: News category (``general``, ``forex``, ``crypto``, ``merger``).
        min_id: Minimum news ID for pagination (use ID from last result).
    """
    return await _get("/news", {"category": category, "minId": min_id})


@finnhub_tool("/company-news")
async def company_news(
    symbol: str,
    from_date: str,
    to_date: str,
) -> dict | list:
    """Get company-specific news articles.

    Args:
        symbol: Stock symbol.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/company-news", {"symbol": symbol, "from": from_date, "to": to_date})


@finnhub_tool("/news-sentiment")
async def news_sentiment(symbol: str) -> dict | list:
    """Get news sentiment and statistics for a company.

    Args:
        symbol: Stock symbol.
    """
    return await _get("/news-sentiment", {"symbol": symbol})


# ===================================================================
# ECONOMIC DATA
# ===================================================================


@finnhub_tool("/country")
async def country() -> dict | list:
    """List supported countries and metadata."""
    return await _get("/country")


@finnhub_tool("/economic/code")
async def economic_code() -> dict | list:
    """List economic indicator codes (used with ``economic_data``)."""
    return await _get("/economic/code")


@finnhub_tool("/economic")
async def economic_data(code: str) -> dict | list:
    """Get economic data for a given indicator code.

    Args:
        code: Economic indicator code (obtain from ``economic_code``).
    """
    return await _get("/economic", {"code": code})


@finnhub_tool("/calendar/economic")
async def calendar_economic(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get economic events calendar.

    Args:
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/calendar/economic", {"from": from_date, "to": to_date})


@finnhub_tool("/calendar/earnings")
async def earnings_calendar(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    symbol: Optional[str] = None,
    international: Optional[bool] = None,
) -> dict | list:
    """Get upcoming and recent earnings dates.

    Args:
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
        symbol: Filter by stock symbol.
        international: Include international companies.
    """
    return await _get(
        "/calendar/earnings",
        {"from": from_date, "to": to_date, "symbol": symbol, "international": international},
    )


@finnhub_tool("/calendar/ipo")
async def ipo_calendar(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get upcoming and recent IPOs.

    Args:
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/calendar/ipo", {"from": from_date, "to": to_date})


# ===================================================================
# INDEX
# ===================================================================


@finnhub_tool("/index/constituents")
async def indices_const(symbol: str) -> dict | list:
    """Get current constituents of a stock index.

    Args:
        symbol: Index symbol (e.g. ``^GSPC`` for S&P 500, ``^DJI`` for Dow Jones).
    """
    return await _get("/index/constituents", {"symbol": symbol})


@finnhub_tool("/index/historical-constituents")
async def indices_hist_const(symbol: str) -> dict | list:
    """Get historical additions and removals from a stock index.

    Args:
        symbol: Index symbol.
    """
    return await _get("/index/historical-constituents", {"symbol": symbol})


# ===================================================================
# ETF
# ===================================================================


@finnhub_tool("/etf/profile")
async def etfs_profile(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
) -> dict | list:
    """Get ETF profile data (expense ratio, AUM, NAV, etc.).

    Args:
        symbol: ETF symbol.
        isin: ISIN identifier.
    """
    return await _get("/etf/profile", {"symbol": symbol, "isin": isin})


@finnhub_tool("/etf/holdings")
async def etfs_holdings(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
    skip: Optional[int] = None,
    date: Optional[str] = None,
) -> dict | list:
    """Get ETF holdings (top positions).

    Args:
        symbol: ETF symbol.
        isin: ISIN identifier.
        skip: Number of records to skip for pagination.
        date: Date for point-in-time holdings (``YYYY-MM-DD``).
    """
    return await _get("/etf/holdings", {"symbol": symbol, "isin": isin, "skip": skip, "date": date})


@finnhub_tool("/etf/sector")
async def etfs_sector_exp(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
) -> dict | list:
    """Get ETF sector exposure breakdown.

    Args:
        symbol: ETF symbol.
        isin: ISIN identifier.
    """
    return await _get("/etf/sector", {"symbol": symbol, "isin": isin})


@finnhub_tool("/etf/country")
async def etfs_country_exp(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
) -> dict | list:
    """Get ETF country exposure breakdown.

    Args:
        symbol: ETF symbol.
        isin: ISIN identifier.
    """
    return await _get("/etf/country", {"symbol": symbol, "isin": isin})


@finnhub_tool("/etf/allocation")
async def etfs_allocation(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
) -> dict | list:
    """Get ETF asset allocation breakdown (equities, bonds, cash, etc.).

    Args:
        symbol: ETF symbol.
        isin: ISIN identifier.
    """
    return await _get("/etf/allocation", {"symbol": symbol, "isin": isin})


# ===================================================================
# MUTUAL FUND
# ===================================================================


@finnhub_tool("/mutual-fund/profile")
async def mutual_fund_profile(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
) -> dict | list:
    """Get mutual fund profile data.

    Args:
        symbol: Fund symbol.
        isin: ISIN identifier.
    """
    return await _get("/mutual-fund/profile", {"symbol": symbol, "isin": isin})


@finnhub_tool("/mutual-fund/holdings")
async def mutual_fund_holdings(
    symbol: Optional[str] = None,
    isin: Optional[str] = None,
    skip: Optional[int] = None,
) -> dict | list:
    """Get mutual fund holdings.

    Args:
        symbol: Fund symbol.
        isin: ISIN identifier.
        skip: Number of records to skip for pagination.
    """
    return await _get("/mutual-fund/holdings", {"symbol": symbol, "isin": isin, "skip": skip})


@finnhub_tool("/mutual-fund/sector")
async def mutual_fund_sector_exp(symbol: str) -> dict | list:
    """Get mutual fund sector exposure.

    Args:
        symbol: Fund symbol.
    """
    return await _get("/mutual-fund/sector", {"symbol": symbol})


@finnhub_tool("/mutual-fund/country")
async def mutual_fund_country_exp(symbol: str) -> dict | list:
    """Get mutual fund country exposure.

    Args:
        symbol: Fund symbol.
    """
    return await _get("/mutual-fund/country", {"symbol": symbol})


@finnhub_tool("/mutual-fund/eet")
async def mutual_fund_eet(isin: str) -> dict | list:
    """Get mutual fund European ESG Template (EET) data.

    Args:
        isin: ISIN identifier.
    """
    return await _get("/mutual-fund/eet", {"isin": isin})


@finnhub_tool("/mutual-fund/eet-pai")
async def mutual_fund_eet_pai(isin: str) -> dict | list:
    """Get mutual fund EET Principal Adverse Impact (PAI) data.

    Args:
        isin: ISIN identifier.
    """
    return await _get("/mutual-fund/eet-pai", {"isin": isin})


# ===================================================================
# BOND
# ===================================================================


@finnhub_tool("/bond/profile")
async def bond_profile(
    isin: Optional[str] = None,
    cusip: Optional[str] = None,
    figi: Optional[str] = None,
) -> dict | list:
    """Get bond profile data (coupon, maturity, issuer, etc.).

    Args:
        isin: ISIN identifier.
        cusip: CUSIP identifier.
        figi: FIGI identifier.
    """
    return await _get("/bond/profile", {"isin": isin, "cusip": cusip, "figi": figi})


@finnhub_tool("/bond/price")
async def bond_price(
    isin: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get bond price data.

    Args:
        isin: ISIN identifier.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/bond/price", {"isin": isin, "from": from_date, "to": to_date})


@finnhub_tool("/bond/tick")
async def bond_tick(
    isin: str,
    date: str,
    limit: Optional[int] = None,
    skip: Optional[int] = None,
    exchange: Optional[str] = None,
) -> dict | list:
    """Get bond tick data.

    Args:
        isin: ISIN identifier.
        date: Date (``YYYY-MM-DD``).
        limit: Maximum number of ticks.
        skip: Number of ticks to skip.
        exchange: Exchange code filter.
    """
    return await _get(
        "/bond/tick",
        {"isin": isin, "date": date, "limit": limit, "skip": skip, "exchange": exchange},
    )


@finnhub_tool("/bond/yield-curve")
async def bond_yield_curve(code: str) -> dict | list:
    """Get yield curve data for a country/region.

    Args:
        code: Yield curve code (e.g. ``10y`` for US 10-year).
    """
    return await _get("/bond/yield-curve", {"code": code})


# ===================================================================
# INSTITUTIONAL
# ===================================================================


@finnhub_tool("/institutional/profile")
async def institutional_profile(cik: str) -> dict | list:
    """Get institutional investor profile by CIK.

    Args:
        cik: CIK number of the institution.
    """
    return await _get("/institutional/profile", {"cik": cik})


@finnhub_tool("/institutional/portfolio")
async def institutional_portfolio(
    cik: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get institutional investor portfolio (13F holdings).

    Args:
        cik: CIK number of the institution.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/institutional/portfolio",
        {"cik": cik, "from": from_date, "to": to_date},
    )


@finnhub_tool("/institutional/ownership")
async def institutional_ownership(
    symbol: str,
    cusip: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get institutional ownership data for a stock.

    Args:
        symbol: Stock symbol.
        cusip: CUSIP identifier.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/institutional/ownership",
        {"symbol": symbol, "cusip": cusip, "from": from_date, "to": to_date},
    )


# ===================================================================
# OTHER
# ===================================================================


@finnhub_tool("/covid19/us")
async def covid19() -> dict | list:
    """Get COVID-19 data for the US (cases by state)."""
    return await _get("/covid19/us")


@finnhub_tool("/fda-advisory-committee-calendar")
async def fda_calendar() -> dict | list:
    """Get FDA advisory committee calendar (PDUFA dates, etc.)."""
    return await _get("/fda-advisory-committee-calendar")


@finnhub_tool("/sector/metrics")
async def sector_metric(region: str) -> dict | list:
    """Get sector-level performance metrics.

    Args:
        region: Region code (e.g. ``NA`` for North America, ``EU`` for Europe).
    """
    return await _get("/sector/metrics", {"region": region})


@finnhub_tool("/ca/symbol-change")
async def symbol_change(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get recent ticker symbol changes (e.g. after M&A or rebranding).

    Args:
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/ca/symbol-change", {"from": from_date, "to": to_date})


@finnhub_tool("/ca/isin-change")
async def isin_change(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get recent ISIN changes.

    Args:
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get("/ca/isin-change", {"from": from_date, "to": to_date})


@finnhub_tool("/airline/price-index")
async def airline_price_index(
    airline: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> dict | list:
    """Get airline ticket price index.

    Args:
        airline: Airline code.
        from_date: Start date (``YYYY-MM-DD``).
        to_date: End date (``YYYY-MM-DD``).
    """
    return await _get(
        "/airline/price-index",
        {"airline": airline, "from": from_date, "to": to_date},
    )


@finnhub_tool("/bank-branch")
async def bank_branch(symbol: str) -> dict | list:
    """Get bank branch data for a financial institution.

    Args:
        symbol: Bank stock symbol.
    """
    return await _get("/bank-branch", {"symbol": symbol})


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------


def main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
