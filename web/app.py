from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from cli.models import AnalystType
from cli.utils import (
    ANALYST_ORDER,
    SHALLOW_AGENT_OPTIONS,
    DEEP_AGENT_OPTIONS,
    LLM_PROVIDER_OPTIONS,
)

app = FastAPI(title="TradingAgents Dashboard", version="0.1.0")

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

PROVIDER_URL_LOOKUP: Dict[str, str] = {
    name.lower(): url for name, url in LLM_PROVIDER_OPTIONS
}
PROVIDER_DISPLAY_LOOKUP: Dict[str, str] = {
    name.lower(): name for name, _ in LLM_PROVIDER_OPTIONS
}

RESEARCH_DEPTH_OPTIONS = [
    {"label": "Shallow · Quick takes (1 round)", "value": 1},
    {"label": "Balanced · Extended dialog (3 rounds)", "value": 3},
    {"label": "Deep Dive · Comprehensive (5 rounds)", "value": 5},
]


def _dates_between(start: datetime, end: datetime) -> List[str]:
    cursor = start
    dates: List[str] = []
    while cursor <= end:
        dates.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=1)
    return dates


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "analysts": [(label, value.value) for label, value in ANALYST_ORDER],
            "research_depth_options": RESEARCH_DEPTH_OPTIONS,
            "provider_options": [
                {"label": name, "value": name.lower(), "url": url}
                for name, url in LLM_PROVIDER_OPTIONS
            ],
            "shallow_models": SHALLOW_AGENT_OPTIONS,
            "deep_models": DEEP_AGENT_OPTIONS,
        },
    )


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(
    request: Request,
    ticker: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    provider: str = Form(...),
    shallow_model: str = Form(...),
    deep_model: str = Form(...),
    research_depth: int = Form(...),
    analysts: List[str] = Form(default=[]),
    risk_profile: str = Form("Balanced"),
    capital_allocation: str = Form(""),
    time_horizon: str = Form("Day Trading"),
    notes: str = Form(""),
) -> HTMLResponse:
    ticker_symbol = ticker.upper().strip()
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "analysts": [(label, value.value) for label, value in ANALYST_ORDER],
                "research_depth_options": RESEARCH_DEPTH_OPTIONS,
                "provider_options": [
                    {"label": name, "value": name.lower(), "url": url}
                    for name, url in LLM_PROVIDER_OPTIONS
                ],
                "shallow_models": SHALLOW_AGENT_OPTIONS,
                "deep_models": DEEP_AGENT_OPTIONS,
                "form_error": "Invalid date format. Use YYYY-MM-DD.",
            },
            status_code=400,
        )

    if end < start:
        start, end = end, start

    selected_analysts = [AnalystType(a) for a in analysts] if analysts else [
        value for _, value in ANALYST_ORDER
    ]

    selected_dates = _dates_between(start, end)

    provider_key = provider.lower()
    backend_url = PROVIDER_URL_LOOKUP.get(provider_key, "https://api.openai.com/v1")

    analysis_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for trade_date in selected_dates:
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = research_depth
        config["max_risk_discuss_rounds"] = research_depth
        config["quick_think_llm"] = shallow_model
        config["deep_think_llm"] = deep_model
        config["backend_url"] = backend_url
        config["llm_provider"] = provider_key

        try:
            graph = TradingAgentsGraph(
                [analyst.value for analyst in selected_analysts],
                debug=False,
                config=config,
            )
            final_state, decision = graph.propagate(ticker_symbol, trade_date)
            analysis_results.append(
                {
                    "date": trade_date,
                    "decision": decision,
                    "market_report": final_state.get("market_report"),
                    "sentiment_report": final_state.get("sentiment_report"),
                    "news_report": final_state.get("news_report"),
                    "fundamentals_report": final_state.get("fundamentals_report"),
                    "investment_plan": final_state.get("investment_plan"),
                    "trader_plan": final_state.get("trader_investment_plan"),
                    "risk_summary": final_state.get("risk_debate_state", {}).get("judge_decision"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            errors.append({"date": trade_date, "message": str(exc)})

    provider_display = PROVIDER_DISPLAY_LOOKUP.get(provider_key, provider.title())

    selected_analyst_labels = [
        label for label, value in ANALYST_ORDER if value in selected_analysts
    ]

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "ticker": ticker_symbol,
            "start_date": selected_dates[0] if selected_dates else start_date,
            "end_date": selected_dates[-1] if selected_dates else end_date,
            "provider": provider_display,
            "backend_url": backend_url,
            "shallow_model": shallow_model,
            "deep_model": deep_model,
            "research_depth": research_depth,
            "analyst_labels": selected_analyst_labels,
            "risk_profile": risk_profile,
            "capital_allocation": capital_allocation,
            "time_horizon": time_horizon,
            "notes": notes,
            "results": analysis_results,
            "errors": errors,
        },
    )