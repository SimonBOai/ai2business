import pytrends
from pytrends.request import TrendReq
# Only need to run this once, the rest of requests will use the same session.

df main():
    pytrend = TrendReq()
    pytrend.build_payload(["Blockchain", "Apple","S&P500"], timeframe='today 5-y')
    df = pytrend.interest_over_time()
    df.plot()

if __name__ == '__main__':
    main()