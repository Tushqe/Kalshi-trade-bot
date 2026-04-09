# Kalshi Trade Bot

A 15-minute BTC scalping bot that trades on [Kalshi](https://kalshi.com) prediction markets using real-time crypto market data.

## How It Works

The bot runs a 10-second loop that:

1. Streams real-time BTC price data via Coinbase WebSocket (1m klines + order book)
2. Evaluates short-term signals (order-book imbalance, trade flow, momentum)
3. Finds the best Kalshi 15-minute BTC contract with sufficient edge
4. Sizes the position using fractional Kelly criterion
5. Places the order on Kalshi

## Architecture

| Module | Description |
|---|---|
| `main.py` | Main event loop and configuration |
| `coinbase_oracle.py` | Coinbase WebSocket feed for real-time BTC data |
| `binance_oracle.py` | Binance WebSocket feed (alternative data source) |
| `short_term_engine.py` | Signal engine combining order-book + trade-flow indicators |
| `signals.py` | Technical signal calculations |
| `strategies.py` | Trading strategy logic |
| `ensemble.py` | Ensemble signal aggregation |
| `router.py` | Kalshi 15m contract selection and routing |
| `risk.py` | Position sizing (Kelly criterion) and risk management |
| `kalshi_client.py` | Kalshi REST API client |
| `kalshi_ws.py` | Kalshi WebSocket client with RSA-PSS authentication |
| `models.py` | Data models (MarketState, TradeInstruction, etc.) |

## Setup

### Prerequisites

- Python 3.12+
- A [Kalshi](https://kalshi.com) account with API access
- An RSA private key generated from Kalshi account settings

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/Kalshi-trade-bot.git
cd Kalshi-trade-bot
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure credentials

Create a `.env` file in the project root (this file is git-ignored and will not be uploaded):

```bash
KALSHI_API_KEY_ID=your-api-key-id
KALSHI_PRIVATE_KEY_PATH=path/to/your/private_key.pem
```

Or export them directly:

```bash
export KALSHI_API_KEY_ID="your-api-key-id"
export KALSHI_PRIVATE_KEY_PATH="path/to/your/private_key.pem"
```

### 5. Run the bot

```bash
python main.py
```

## Configuration

All parameters can be tuned via environment variables:

| Variable | Default | Description |
|---|---|---|
| `KALSHI_API_KEY_ID` | *(required)* | Kalshi API key ID |
| `KALSHI_PRIVATE_KEY_PATH` | *(required)* | Path to RSA private key PEM file |
| `LOOP_INTERVAL` | `10` | Seconds between evaluation cycles |
| `INITIAL_EQUITY_CENTS` | `9730` | Starting equity in cents |
| `MIN_EDGE` | `0.08` | Minimum edge threshold (8%) |
| `MAX_ALLOC_PCT` | `0.05` | Max allocation per trade (5%) |
| `KELLY_FRAC` | `0.25` | Kelly fraction for position sizing |
| `MAX_DD_PCT` | `0.10` | Max daily drawdown before halt (10%) |
| `MAX_OPEN_POS` | `3` | Max concurrent open positions |

## Security

- **Never commit your private key or `.env` file.** The `.gitignore` is configured to exclude `*.pem`, `*.key`, `Boroldoi.txt`, and `.env`.
- Store your RSA private key outside the project directory when possible.
- The bot uses RSA-PSS signing for Kalshi API authentication — your private key never leaves your machine.

## Running Tests

```bash
python -m pytest test_audit_fixes.py -v
```

## Disclaimer

This bot is for educational and personal use. Trading on prediction markets involves financial risk. Use at your own discretion.
