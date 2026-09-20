"""MCP server exposing live travel information as tools.

Two capabilities, three tools, no API keys required:

  * get_current_weather    — Open-Meteo current conditions
  * get_weather_forecast   — Open-Meteo daily forecast (up to 16 days)
  * convert_currency       — Frankfurter (European Central Bank reference rates)

Every tool returns a dict with a "status" field. On failure it returns
status="error" with a message instead of raising, so the assistant can report
"the tool was unavailable" rather than inventing a number.

Run standalone (for the MCP Inspector):
    python -m mcp_server.travel_tools
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("travel-tools")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Frankfurter moved from api.frankfurter.app to api.frankfurter.dev/v1;
# the old host now 301-redirects. Use the current host and follow redirects.
FX_URL = "https://api.frankfurter.dev/v1/latest"
TIMEOUT = 15.0

# Known coordinates avoid a geocoding round-trip for the primary destination.
KNOWN_PLACES: dict[str, tuple[float, float, str]] = {
    "singapore": (1.3521, 103.8198, "Singapore"),
    "sentosa": (1.2494, 103.8303, "Sentosa, Singapore"),
    "delhi": (28.6139, 77.2090, "Delhi, India"),
    "new delhi": (28.6139, 77.2090, "New Delhi, India"),
    "mumbai": (19.0760, 72.8777, "Mumbai, India"),
}

WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snowfall",
    73: "moderate snowfall",
    75: "heavy snowfall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _error(message: str, tool: str) -> dict[str, Any]:
    return {"status": "error", "tool": tool, "message": message, "retrieved_at": _now()}


def _describe(code: Any) -> str:
    try:
        return WMO_CODES.get(int(code), f"weather code {code}")
    except (TypeError, ValueError):
        return "unknown conditions"


def _geocode(location: str) -> tuple[float, float, str]:
    key = location.strip().lower()
    if key in KNOWN_PLACES:
        return KNOWN_PLACES[key]

    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(GEOCODE_URL, params={"name": location, "count": 1})
        resp.raise_for_status()
        results = resp.json().get("results") or []

    if not results:
        raise ValueError(f"could not find coordinates for '{location}'")

    top = results[0]
    label = ", ".join(
        part for part in [top.get("name"), top.get("country")] if part
    )
    return float(top["latitude"]), float(top["longitude"]), label


@mcp.tool()
def get_current_weather(location: str = "Singapore") -> dict[str, Any]:
    """Get the current weather conditions for a city.

    Use for questions about the weather right now. Returns temperature in
    Celsius, humidity, precipitation and a text description.
    """
    try:
        lat, lon, label = _geocode(location)
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": (
                        "temperature_2m,relative_humidity_2m,apparent_temperature,"
                        "precipitation,weather_code,wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        current = data.get("current", {})
        return {
            "status": "ok",
            "tool": "get_current_weather",
            "source": "Open-Meteo API",
            "location": label,
            "observed_at": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "feels_like_c": current.get("apparent_temperature"),
            "humidity_percent": current.get("relative_humidity_2m"),
            "precipitation_mm": current.get("precipitation"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "conditions": _describe(current.get("weather_code")),
            "retrieved_at": _now(),
        }
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), "get_current_weather")


@mcp.tool()
def get_weather_forecast(location: str = "Singapore", days: int = 3) -> dict[str, Any]:
    """Get a day-by-day weather forecast for a city.

    Use for trip planning and for deciding between indoor and outdoor
    activities. `days` may be 1-16. Each day includes max/min temperature,
    total precipitation, rain probability and a text description.
    """
    try:
        days = max(1, min(int(days), 16))
        lat, lon, label = _geocode(location)
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(
                FORECAST_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": (
                        "weather_code,temperature_2m_max,temperature_2m_min,"
                        "precipitation_sum,precipitation_probability_max"
                    ),
                    "forecast_days": days,
                    "timezone": "auto",
                },
            )
            resp.raise_for_status()
            daily = resp.json().get("daily", {})

        dates = daily.get("time", [])
        out = []
        for i, date in enumerate(dates):
            rain_chance = (daily.get("precipitation_probability_max") or [None])[i]
            rain_mm = (daily.get("precipitation_sum") or [None])[i]
            out.append(
                {
                    "date": date,
                    "conditions": _describe((daily.get("weather_code") or [None])[i]),
                    "temp_max_c": (daily.get("temperature_2m_max") or [None])[i],
                    "temp_min_c": (daily.get("temperature_2m_min") or [None])[i],
                    "precipitation_mm": rain_mm,
                    "rain_probability_percent": rain_chance,
                    "outdoor_suitability": _suitability(rain_chance, rain_mm),
                }
            )

        return {
            "status": "ok",
            "tool": "get_weather_forecast",
            "source": "Open-Meteo API",
            "location": label,
            "days": len(out),
            "forecast": out,
            "retrieved_at": _now(),
        }
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), "get_weather_forecast")


def _suitability(rain_chance: Any, rain_mm: Any) -> str:
    try:
        chance = float(rain_chance) if rain_chance is not None else 0.0
        millimetres = float(rain_mm) if rain_mm is not None else 0.0
    except (TypeError, ValueError):
        return "unknown"
    if chance >= 70 or millimetres >= 10:
        return "poor — plan indoor activities"
    if chance >= 40 or millimetres >= 2:
        return "mixed — keep an indoor backup"
    return "good — outdoor activities fine"


@mcp.tool()
def convert_currency(
    amount: float, from_currency: str, to_currency: str
) -> dict[str, Any]:
    """Convert an amount of money from one currency to another.

    Uses European Central Bank reference rates. Currency codes are ISO 4217,
    e.g. INR, SGD, USD, EUR. Use for any budget or price conversion question.
    """
    try:
        amount = float(amount)
        base = str(from_currency).strip().upper()[:3]
        target = str(to_currency).strip().upper()[:3]

        if amount <= 0:
            return _error("amount must be greater than zero", "convert_currency")
        if base == target:
            return {
                "status": "ok",
                "tool": "convert_currency",
                "source": "no conversion needed",
                "amount": amount,
                "from_currency": base,
                "to_currency": target,
                "rate": 1.0,
                "converted_amount": round(amount, 2),
                "rate_date": None,
                "retrieved_at": _now(),
            }

        with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
            resp = client.get(
                FX_URL, params={"amount": amount, "from": base, "to": target}
            )
            if resp.status_code == 404:
                return _error(
                    f"currency pair {base}->{target} is not supported by the "
                    "exchange-rate service",
                    "convert_currency",
                )
            resp.raise_for_status()
            data = resp.json()

        rates = data.get("rates") or {}
        if target not in rates:
            return _error(
                f"no rate returned for {base}->{target}", "convert_currency"
            )

        converted = float(rates[target])
        return {
            "status": "ok",
            "tool": "convert_currency",
            "source": "Frankfurter / European Central Bank reference rates",
            "amount": amount,
            "from_currency": base,
            "to_currency": target,
            "converted_amount": round(converted, 2),
            "rate": round(converted / amount, 6) if amount else None,
            "rate_date": data.get("date"),
            "retrieved_at": _now(),
        }
    except Exception as exc:  # noqa: BLE001
        return _error(str(exc), "convert_currency")


if __name__ == "__main__":
    mcp.run()
