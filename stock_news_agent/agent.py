from google.adk.agents.llm_agent import Agent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
import os
import requests
from datetime import datetime, timedelta

load_dotenv()

OPENAI_MODEL = LiteLlm(model="openai/gpt-4o-mini")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


def fetch_stock_news(ticker: str) -> dict:
    """Fetch recent company news from Finnhub for the last 3 days."""
    end_date = datetime.today()
    start_date = end_date - timedelta(days=3)

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from": start_date.strftime("%Y-%m-%d"),
        "to": end_date.strftime("%Y-%m-%d"),
        "token": FINNHUB_API_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    articles = []
    for item in data[:5]:
        articles.append({
            "headline": item.get("headline"),
            "summary": item.get("summary"),
            "source": item.get("source"),
            "url": item.get("url"),
        })

    return {
        "ticker": ticker.upper(),
        "news_count": len(articles),
        "articles": articles,
    }


def fetch_market_context(ticker: str) -> dict:
    """Fetch recent stock price context from Alpha Vantage."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker.upper(),
        "apikey": ALPHA_VANTAGE_API_KEY,
    }

    response = requests.get(url, params=params)
    data = response.json()

    time_series = data.get("Time Series (Daily)", {})
    recent_dates = list(time_series.keys())[:5]

    prices = []
    for date in recent_dates:
        day = time_series[date]
        prices.append({
            "date": date,
            "open": day.get("1. open"),
            "high": day.get("2. high"),
            "low": day.get("3. low"),
            "close": day.get("4. close"),
            "volume": day.get("5. volume"),
        })

    return {
        "ticker": ticker.upper(),
        "recent_price_data": prices,
    }


def predict_stock_movement(ticker: str, sentiment_score: int, volatility_score: int) -> dict:
    """Simple placeholder prediction model using sentiment and volatility."""
    rise_score = min(10, max(0, round((sentiment_score * 0.7) + ((10 - volatility_score) * 0.3))))
    drop_score = 10 - rise_score

    if rise_score >= 7:
        recommendation = "Watch / Potential Upside"
    elif drop_score >= 7:
        recommendation = "Avoid / High Risk"
    else:
        recommendation = "Neutral / Monitor"

    return {
        "ticker": ticker.upper(),
        "rise_likelihood": rise_score,
        "drop_likelihood": drop_score,
        "recommendation": recommendation,
    }


data_retrieval_agent = Agent(
    name="data_retrieval_agent",
    model=OPENAI_MODEL,
    description="Fetches recent stock news and stock price data.",
    instruction="""
    You are the Data Retrieval Agent.
    Your job is to fetch recent financial news and market data for the ticker provided by the user.
    Use fetch_stock_news and fetch_market_context.
    Return only structured information.
    """,
    tools=[fetch_stock_news, fetch_market_context],
)


news_analysis_agent = Agent(
    name="news_analysis_agent",
    model=OPENAI_MODEL,
    description="Analyzes financial news sentiment and extracts key events.",
    instruction="""
    You are the News Analysis Agent.
    Analyze the news headlines and summaries.
    Identify sentiment, key drivers, and event types such as earnings, analyst ratings,
    macroeconomic news, product updates, lawsuits, partnerships, or management changes.

    Return:
    sentiment_score: 0-10
    sentiment_label: Positive / Neutral / Negative
    key_events:
    rationale:
    """,
)


market_context_agent = Agent(
    name="market_context_agent",
    model=OPENAI_MODEL,
    description="Evaluates recent price trend, volatility, and market context.",
    instruction="""
    You are the Market Context Agent.
    Review recent stock price data and explain:
    - short-term trend
    - volatility level from 0-10
    - unusual price or volume movement
    - whether recent price movement appears bullish, bearish, or neutral
    """,
)


prediction_agent = Agent(
    name="prediction_agent",
    model=OPENAI_MODEL,
    description="Predicts stock rise/drop likelihood using sentiment and market context.",
    instruction="""
    You are the Prediction Agent.
    Use the news sentiment and market context to estimate whether the stock is more likely
    to rise or drop.

    Use predict_stock_movement when sentiment_score and volatility_score are available.
    Return scores from 0-10.
    """,
    tools=[predict_stock_movement],
)


recommendation_agent = Agent(
    name="recommendation_agent",
    model=OPENAI_MODEL,
    description="Creates final user-friendly investment insight.",
    instruction="""
    You are the Recommendation Agent.
    Convert the prediction into a clear final response.

    Format:
    Stock:
    News Sentiment:
    Market Context:
    Rise Likelihood (0-10):
    Drop Likelihood (0-10):
    Key Influencing Factors:
    Recommendation:
    Disclaimer: This is not financial advice.
    """,
)


root_agent = Agent(
    name="stock_news_prediction_orchestrator",
    model=OPENAI_MODEL,
    description="Orchestrates a multi-agent stock news prediction workflow.",
    instruction="""
    You are the Orchestrator Agent.

    When the user provides a stock ticker:
    1. Ask Data Retrieval Agent to fetch recent news and price data.
    2. Ask News Analysis Agent to analyze sentiment and key events.
    3. Ask Market Context Agent to analyze trend and volatility.
    4. Ask Prediction Agent to estimate rise/drop likelihood.
    5. Ask Recommendation Agent to create the final response.

    Keep the final answer structured and easy to understand.
    Always include a disclaimer that this is not financial advice.
    """,
    sub_agents=[
        data_retrieval_agent,
        news_analysis_agent,
        market_context_agent,
        prediction_agent,
        recommendation_agent,
    ],
)