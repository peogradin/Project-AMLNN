import argparse
import time
import yfinance as yf
import pandas as pd


def fetch_omxs30(period: str = '10y', interval: str = '1d', retries: int = 5, backoff_factor: float = 1.5) -> pd.DataFrame:
    """
    Fetches historical data for the OMXS30 index from Yahoo Finance, with retry logic on rate limits.

    Args:
        period (str): Data period to download (e.g., '1d', '5d', '1mo', '1y', '10y', 'max').
        interval (str): Data interval (e.g., '1m', '5m', '1h', '1d').
        retries (int): Number of retry attempts on errors.
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
        except Exception as e:
            msg = str(e).lower()
            # Retry only on rate limit or service errors
            if 'rate limit' not in msg and '503' not in msg:
                raise
            if attempt == retries:
                print(f"Exceeded retry limit: {e}")
                raise
            sleep_time = wait * (backoff_factor ** attempt)
            print(f"Rate limit or service error detected, retrying in {sleep_time:.1f} seconds... ({e})")
            time.sleep(sleep_time)
            attempt += 1

    # Final attempt
    return ticker.history(period=period, interval=interval)


def main():
    parser = argparse.ArgumentParser(
        description='Fetch historical data for the OMXS30 index from Yahoo Finance and save to CSV.'
    )
    parser.add_argument('--period', type=str, default='10y',
                        help="Data period to download (e.g., '1d', '5d', '1mo', '1y', '10y', 'max').")
    parser.add_argument('--interval', type=str, default='1d',
                        help="Data interval (e.g., '1m', '5m', '1h', '1d').")
    parser.add_argument('--retries', type=int, default=5,
                        help="Number of retry attempts on errors.")
    parser.add_argument('--backoff', type=float, default=1.5,
                        help="Backoff multiplier for retry delays.")
    parser.add_argument('--outfile', type=str, default='omxs30_10y.csv',
                        help="Path to output CSV file.")
    args = parser.parse_args()

    try:
        data = fetch_omxs30(
            period=args.period,
            interval=args.interval,
            retries=args.retries,
            backoff_factor=args.backoff
        )
        if data.empty:
            print("No data fetched. CSV not created.")
        else:
            data.to_csv(args.outfile)
            print(f"Data for period {args.period} @ interval {args.interval} saved to {args.outfile}")
    except Exception as e:
        print(f"Failed to fetch or save data: {e}")


if __name__ == '__main__':
    main()
