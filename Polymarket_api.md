Using Polymarket APIs for Crypto Data and Prediction‑Market Analysis
Introduction

Polymarket is a prediction‑market platform that lets traders buy or sell outcome tokens representing real‑world events (e.g., “Will Bitcoin close above $40k on 31 December 2025?”). Unlike a spot exchange, the underlying price reflects the probability of the event occurring rather than a continuous asset price. Polymarket operates a Central Limit Order Book (CLOB) for each market and exposes several APIs for accessing market data, order books and real‑time crypto prices. It also runs a Gamma API for market discovery and a real‑time data stream (RTDS) for streaming cryptocurrency prices via WebSocket.

This guide explains how to retrieve Polymarket data relevant to crypto trading and how to use it alongside traditional crypto price data (e.g., from Binance) for statistical arbitrage and trend‑following strategies.

API overview
Base endpoints

Polymarket exposes multiple APIs:

Service	Purpose	Base endpoint
CLOB REST API	Order book, pricing and trade endpoints for outcome tokens.	https://clob.polymarket.com
Gamma Markets API	Market discovery; returns events, markets, tags and metadata.	https://gamma-api.polymarket.com
WebSocket (CLOB)	Real‑time order and trade updates.	wss://ws-subscriptions-clob.polymarket.com/ws/
Real‑Time Data Stream (RTDS)	Real‑time cryptocurrency price updates from Binance and Chainlink.	wss://ws-live-data.polymarket.com
docs.polymarket.com

Data‑API	User data (holdings, positions). Requires authentication and is outside the scope of this overview.	https://data-api.polymarket.com

Timestamps in the REST API are returned as Unix time (seconds or milliseconds) and should be converted to your local timezone when displaying or analysing the data.

Authentication and rate limits

Most CLOB pricing endpoints are public and do not require authentication. Order‑manipulation endpoints (placing/cancelling orders) and the trades endpoint require an L2 header (API key) for authentication. WebSocket connections to the CLOB use an auth field along with market/asset IDs to subscribe to user‑specific or market‑wide events
docs.polymarket.com
. Rate limits are documented separately; always respect them and implement retry logic.

Discovering markets via the Gamma API

Before you can fetch pricing or order‑book data you must identify the market and its token IDs. Each market consists of two tokens—one for each outcome (e.g., “YES” and “NO”). The Gamma API helps you find markets and their token IDs.

Retrieval strategies

The Gamma guide outlines three ways to fetch markets
docs.polymarket.com
:

By slug: If you know a specific market’s slug (e.g., fed-decision-in-october from the URL), call /events/slug/<slug> or /markets/slug/<slug> on gamma-api.polymarket.com
docs.polymarket.com
.

By tags: Use /tags or /sports to discover tags and filter markets by category. Pass tag_id and closed=false to /markets or /events to retrieve active markets
docs.polymarket.com
. This is useful for finding crypto‑themed markets (e.g., tags related to Bitcoin or cryptocurrency regulation).

All active markets: Use /events?closed=false or /markets?closed=false with pagination parameters (limit, offset)
docs.polymarket.com
. Set order=id and ascending=false to get the newest events first.

A market object returned from /markets includes fields such as condition_id, question_id, tokens (two tokens with token_id and outcome), category, end date, slug and other metadata
docs.polymarket.com
. Save the token_id of the outcome you want to trade or analyse; you will need it to query pricing and order‑book endpoints.

CLOB pricing and order‑book endpoints

Once you have a token ID, you can query the CLOB REST API for price and liquidity information. The following endpoints operate on https://clob.polymarket.com and do not require authentication unless noted.

Order book summary

Endpoint: GET /book

Parameters: token_id (string) – the ERC‑1155 token identifier.

Description: Returns a snapshot of the order book for a token, including lists of bid and ask levels. The response also includes market (condition ID), asset_id (token ID), timestamp, min_order_size, neg_risk flag (whether the market is negative‑risk), tick_size, and arrays of bids and asks with price and size
docs.polymarket.com
. This endpoint helps gauge liquidity and depth.

Example response excerpt:

{
  "market": "0x…",                // condition ID
  "asset_id": "…",                 // token ID
  "timestamp": "2023-10-01T12:00:00Z",
  "min_order_size": "0.001",
  "tick_size": "0.01",
  "neg_risk": false,
  "bids": [{"price":"0.45", "size":"150"}, …],
  "asks": [{"price":"0.47", "size":"120"}, …]
}

Best bid/ask price (market price)

Endpoint: GET /price

Parameters:

token_id (string) – token identifier.

side (enum BUY or SELL) – whether you want the best ask (BUY) or best bid (SELL).

Description: Returns the best limit price on the specified side
docs.polymarket.com
. Useful for quickly obtaining the market price (probability) of an outcome.

Example request: GET /price?token_id=<tokenId>&side=BUY → { "price": "0.45" }.

Multiple prices

Endpoint: POST /prices

Request body: A list of objects with token_id and side (BUY/SELL).

Description: Returns best prices for multiple tokens in a single call. The response is a dictionary keyed by asset_id with nested side/price values
docs.polymarket.com
.

Example request body:

{
  "params": [
    {"token_id": "<tokenA>", "side": "BUY"},
    {"token_id": "<tokenA>", "side": "SELL"},
    {"token_id": "<tokenB>", "side": "BUY"}
  ]
}

Midpoint price

Endpoint: GET /midpoint

Parameters: token_id (string).

Description: Returns the midpoint between the best bid and best ask price
docs.polymarket.com
. Midpoint is often used as the fair probability for an event.

Price history

Endpoint: GET /prices-history

Parameters:

market (string) – CLOB token ID.

startTs and endTs (Unix timestamps) – optional; specify a date range.

interval (enum) – mutually exclusive with startTs/endTs. Supported values include 1m, 1h, 6h, 1d, 1w, max
docs.polymarket.com
.

fidelity (number) – resolution in minutes.

Description: Returns a list of objects with t (timestamp) and p (midpoint price)
docs.polymarket.com
. Use this endpoint to build a price time series for computing technical indicators. For example, retrieving 1‑minute price history and then averaging it over longer windows.

Example response:

{
  "history": [
    {"t": 1697875200, "p": 0.45},
    {"t": 1697875260, "p": 0.451},
    …
  ]
}

Bid‑ask spread

Endpoint: GET /spread (single token) or GET /spreads (multiple tokens).

Parameters:

token_id for /spread.

For /spreads, the request body contains a list of BookParams with token_id values.

Description: Returns the difference between the best ask and best bid price for each token. The spread indicates liquidity and slippage costs. Single‑token response: { "spread": "0.013" }
docs.polymarket.com
. Multi‑token response: { [asset_id]: spread }
docs.polymarket.com
.

Trades

The GET /data/trades endpoint returns trade history for an authenticated user. It requires an L2 header and filters like id, taker, maker, market, before, and after. Response includes trade ID, price, size, status, match time and maker order details
docs.polymarket.com
. This is useful if you plan to track your own executions; public trade data for all traders is not provided through this endpoint.

Real‑Time Data Stream (RTDS) for cryptocurrency prices

Polymarket provides a separate WebSocket service for real‑time cryptocurrency price updates. This is particularly relevant if you want to compare Polymarket’s event‑probability markets against actual crypto spot prices.

WebSocket endpoint: wss://ws-live-data.polymarket.com
docs.polymarket.com
.

Topics:

crypto_prices – streaming prices from Binance.

crypto_prices_chainlink – streaming prices from Chainlink oracles
docs.polymarket.com
.

Subscription message:

{
  "action": "subscribe",
  "subscriptions": [
    {"topic": "crypto_prices", "type": "update", "filters": "btcusdt,ethusdt"}
  ]
}


To subscribe to specific symbols, include a comma‑separated filters field (Binance symbols use lowercase concatenated pairs like btcusdt; Chainlink symbols use slash format like eth/usd)
docs.polymarket.com
.

Message format: Each update includes topic, type, timestamp and a payload containing symbol, timestamp and value (price)
docs.polymarket.com
. Example:

{
  "topic": "crypto_prices",
  "type": "update",
  "timestamp": 1753314064237,
  "payload": {
    "symbol": "btcusdt",
    "timestamp": 1753314064213,
    "value": 67234.50
  }
}


The RTDS service does not require authentication and is suitable for low‑latency monitoring of crypto spot prices to build arbitrage signals. Use this stream alongside Binance’s own WebSocket or REST data for redundancy.

WebSocket channels for the CLOB

For real‑time updates on Polymarket orders and trades, you can use the CLOB WebSocket channel.

Endpoint: wss://ws-subscriptions-clob.polymarket.com/ws/
docs.polymarket.com
.

Channels: USER (user‑specific order/trade events) and MARKET (market‑wide updates). Subscriptions require an auth object and arrays of markets (condition IDs) or asset_ids (token IDs) to specify which events you wish to receive
docs.polymarket.com
.

For market data analysis, the MARKET channel can push updates whenever there is a new order or trade on specified tokens. However, for most statistical analysis you can rely on REST endpoints and the RTDS feed.

Computing indicators with Polymarket price data

Polymarket’s price data differs from traditional OHLCV time series: it represents probabilities of event outcomes and is a single price per token. Nevertheless, you can compute indicators on the time series of probabilities to detect trends or anomalies. Use the /prices-history endpoint to build a price series and then calculate rolling averages, momentum or z‑scores. For example:

import requests
import pandas as pd

# fetch 1‑minute price history for a token
params = {
    'market': '0x...tokenId...',
    'interval': '1h'  # or provide startTs and endTs
}
resp = requests.get('https://clob.polymarket.com/prices-history', params=params)
data = resp.json()['history']

# convert to DataFrame
prices = pd.DataFrame(data)
prices['t'] = pd.to_datetime(prices['t'], unit='s')
prices.set_index('t', inplace=True)

# compute 20‑period moving average and z‑score of price
prices['ma_20'] = prices['p'].rolling(window=20).mean()
prices['std_20'] = prices['p'].rolling(window=20).std()
prices['zscore'] = (prices['p'] - prices['ma_20']) / prices['std_20']


Because Polymarket price series range between 0 and 1, the volatility is usually lower than crypto spot prices. When combining this data with Binance price data (e.g., via the RTDS feed), you might look for mispricings between the implied probability of an event (e.g., “Bitcoin will be above $50k by 31 Dec 2025”) and the probability implied by your own statistical model of Bitcoin’s price path. Significant deviations could present statistical arbitrage opportunities.

Using Polymarket data for crypto‑focused strategies

Identify relevant markets: Use the Gamma API’s tag filtering to find markets about cryptocurrency prices or regulatory events. For example, search tags for “crypto”, “BTC” or “ETH”, then call /markets with tag_id=... and closed=false to retrieve active markets
docs.polymarket.com
. Extract the token_id for the desired outcome (often the “YES” token if you want to bet the event occurs).

Measure market liquidity: For each token_id, call /book to inspect the order book depth and /spread to understand bid‑ask spread
docs.polymarket.com
docs.polymarket.com
. Thin books and wide spreads indicate low liquidity and higher transaction costs.

Collect historical probabilities: Use /prices-history to obtain time‑series data on the token’s probability
docs.polymarket.com
. Compute moving averages, z‑scores or other indicators to detect trends or mean‑reverting behaviours similar to statistical arbitrage on continuous prices.

Compare with spot prices: Use the RTDS crypto_prices stream to retrieve current spot prices from Binance
docs.polymarket.com
. Build a model to transform spot price into a probability of the event (e.g., using option‑pricing techniques or logistic regression). Differences between Polymarket probability and your model’s probability can signal mispricing.

Monitor funding and sentiment: Although Polymarket markets don’t have funding rates like perpetual futures, you can monitor crypto sentiment through open interest and funding rates on exchanges like Binance. Combine this information with Polymarket probability trends to refine your trading signals.

Best practices and caveats

Token IDs vs. market slugs: Always map the token_id back to its human‑readable market via the Gamma API. A market has two tokens; ensure you choose the correct outcome (YES or NO) for your analysis
docs.polymarket.com
.

Liquidity risk: Many Polymarket markets have low depth and wide spreads. Use the spread and book endpoints to gauge transaction cost. Avoid markets with extremely wide spreads.

Time zones: Polymarket uses Unix timestamps (seconds); convert them to Europe/London time when analysing.

No volume data: Price history includes only the midpoint price; it lacks volume information. Use the number of orders in the order book as a proxy for activity, but recognise that statistical indicators like RSI based on volume may not be applicable.

Regulatory considerations: Polymarket markets may be subject to regulatory restrictions depending on your jurisdiction. Always verify that you can participate legally.

Conclusion

Polymarket’s APIs allow developers to retrieve order‑book data, price snapshots, midpoint prices, price history and bid‑ask spreads for prediction markets. Combined with the Gamma API for market discovery and the RTDS crypto price feed, these tools provide a foundation for analysing event probabilities and designing arbitrage strategies. While Polymarket price data represents probabilities rather than continuous asset prices, you can still compute trends and mean‑reversion indicators on the probability series and compare them with external crypto price signals. Use the endpoints summarised above and integrate them with Python data‑analysis libraries to build your own crypto‑focused trading models.