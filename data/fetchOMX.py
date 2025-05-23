import argparse
import time
import yfinance as yf
from yfinance.utils import YFRateLimitError


def fetch_omxs30(period: str = '1d', interval: str = '1d', retries: int = 5, backoff_factor: float = 1.5) -> "pandas.DataFrame":
    """
    Fetches historical data for the OMXS30 index from Yahoo Finance, with retry logic on rate limits.

    Args:
        period (str): Data period to download (e.g., '1d', '5d', '1mo', '1y', 'max').
        interval (str): Data interval (e.g., '1m', '5m', '1h', '1d').
        retries (int): Number of retry attempts on rate limiting.
        backoff_factor (float): Multiplier for sleep time between retries.

    Returns:
        pandas.DataFrame: Historical market data for the OMXS30 index.
    """
    ticker_symbol = '^OMX'
    ticker = yf.Ticker(ticker_symbol)
    attempt = 0
    wait = 1.0

    while attempt <= retries:
        try:
            hist = ticker.history(period=period, interval=interval)
            return hist
        except YFRateLimitError:
            if attempt == retries:
                raise
            sleep_time = wait * (backoff_factor ** attempt)
            print(f"Rate limit hit, retrying in {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            attempt += 1

    # Fallback empty DataFrame if somehow no data
    return ticker.history(period=period, interval=interval)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch historical data for the OMXS30 index from Yahoo Finance with retry on rate limits.'
    )
    parser.add_argument(
        '--period', type=str, default='1d',
        help="Data period to download (e.g., '1d', '5d', '1mo', '1y', 'max')."
    )
    parser.add_argument(
        '--interval', type=str, default='1d',
        help="Data interval (e.g., '1m', '5m', '1h', '1d')."
    )
    parser.add_argument(
        '--retries', type=int, default=5,
        help="Number of retry attempts on rate limiting."
    )
    parser.add_argument(
        '--backoff', type=float, default=1.5,
        help="Backoff multiplier for retry delays."
    )
    args = parser.parse_args()

    try:
        data = fetch_omxs30(
            period=args.period,
            interval=args.interval,
            retries=args.retries,
            backoff_factor=args.backoff
        )
        print(data)
    except YFRateLimitError:
        print("Failed to fetch data due to repeated rate limits. Please try again later.")


if __name__ == '__main__':
    main()
