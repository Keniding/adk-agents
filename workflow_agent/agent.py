"""
travel_planner — ejemplo multi-agente con ADK 2.x (Task API).

Coordinador (chat) que delega en:
  - weather_checker  -> mode="single_turn": nunca habla con el usuario, responde en un solo ciclo.
  - flight_booker    -> mode="task": puede hacer preguntas al usuario hasta completar la reserva
                        y termina llamando a finish_task (inyectado por ADK) con un FlightResult.

Todas las tools están mockeadas para que la demo sea determinista.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Literal

from google.adk import Agent
from google.adk.tools import FunctionTool, ToolContext
from pydantic import BaseModel, Field

MODEL = "gemini-3.5-flash-lite"


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------

class WeatherReport(BaseModel):
    """Salida estructurada del weather_checker."""
    city: str
    latitude: float
    longitude: float
    condition: str = Field(description="Ej. 'soleado', 'nublado', 'lluvia'")
    temperature_c: float
    travel_advice: str = Field(description="Recomendación breve para el viajero")


class FlightInput(BaseModel):
    """Lo que el coordinador le pasa al flight_booker."""
    origin: str = Field(description="Código IATA de origen, ej. 'LIM'")
    destination: str = Field(description="Código IATA de destino, ej. 'CUZ'")
    departure_date: str = Field(
        description="Fecha de salida en formato YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    passengers: int = Field(default=1, ge=1, le=9)
    cabin: Literal["economy", "premium", "business"] = "economy"

class FlightOption(BaseModel):
    flight_id: str
    airline: str
    departure: str
    arrival: str
    price_usd: float


class FlightResult(BaseModel):
    """Payload que flight_booker debe entregar en finish_task."""
    status: Literal["booked", "cancelled", "no_availability"]
    booking_reference: str | None = None
    flight: FlightOption | None = None
    passenger_name: str | None = None
    total_price_usd: float | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Tools del weather_checker
# ---------------------------------------------------------------------------

_GEO_DB = {
    "lima": (-12.0464, -77.0428),
    "cusco": (-13.5320, -71.9675),
    "arequipa": (-16.4090, -71.5375),
    "piura": (-5.1945, -80.6328),
}

_WEATHER_DB = {
    "lima": ("nublado", 18.0),
    "cusco": ("soleado", 21.0),
    "arequipa": ("soleado", 24.0),
    "piura": ("soleado", 31.0),
}


def geocode_address(address: str) -> dict:
    """Convierte una ciudad o dirección en coordenadas geográficas.

    Args:
        address: Nombre de la ciudad o dirección, ej. 'Cusco'.

    Returns:
        dict con status, latitude y longitude, o un error_message.
    """
    key = address.strip().lower()
    for city, (lat, lng) in _GEO_DB.items():
        if city in key:
            return {"status": "success", "city": city.title(), "latitude": lat, "longitude": lng}
    return {"status": "error", "error_message": f"No se encontraron coordenadas para '{address}'."}


def get_weather(latitude: float, longitude: float) -> dict:
    """Obtiene el clima actual para unas coordenadas.

    Args:
        latitude: Latitud obtenida con geocode_address.
        longitude: Longitud obtenida con geocode_address.

    Returns:
        dict con status, condition y temperature_c, o un error_message.
    """
    for city, (lat, lng) in _GEO_DB.items():
        if abs(lat - latitude) < 0.1 and abs(lng - longitude) < 0.1:
            condition, temp = _WEATHER_DB[city]
            return {"status": "success", "condition": condition, "temperature_c": temp}
    return {"status": "error", "error_message": "Sin datos de clima para esas coordenadas."}


def user_info(tool_context: ToolContext) -> dict:
    """Devuelve el perfil del usuario actual (nombre, ciudad de origen, preferencias).

    Returns:
        dict con el perfil del usuario.
    """
    # Lee del estado de sesión si existe; si no, usa un perfil mock.
    state = tool_context.state
    return {
        "status": "success",
        "name": state.get("user:name", "Keniding Tarazona"),
        "home_city": state.get("user:home_city", "Lima"),
        "preferred_cabin": state.get("user:preferred_cabin", "economy"),
        "temperature_unit": "celsius",
    }


# ---------------------------------------------------------------------------
# Tools del flight_booker
# ---------------------------------------------------------------------------

def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    passengers: int = 1,
    cabin: str = "economy",
) -> dict:
    """Busca vuelos disponibles.

    Args:
        origin: Código IATA de origen, ej. 'LIM'.
        destination: Código IATA de destino, ej. 'CUZ'.
        departure_date: Fecha en formato YYYY-MM-DD.
        passengers: Número de pasajeros.
        cabin: 'economy', 'premium' o 'business'.

    Returns:
        dict con status y una lista de opciones de vuelo.
    """
    multiplier = {"economy": 1.0, "premium": 1.6, "business": 2.8}.get(cabin, 1.0)
    base = [
        ("LA2021", "LATAM", "06:10", "07:35", 89.0),
        ("H24410", "Sky Airline", "09:45", "11:05", 72.0),
        ("JA7101", "JetSMART", "14:20", "15:40", 65.0),
    ]
    options = [
        {
            "flight_id": fid,
            "airline": airline,
            "departure": f"{departure_date} {dep}",
            "arrival": f"{departure_date} {arr}",
            "price_usd": round(price * multiplier * passengers, 2),
        }
        for fid, airline, dep, arr, price in base
    ]
    return {"status": "success", "origin": origin.upper(), "destination": destination.upper(), "options": options}


def book_flight(flight_id: str, passenger_name: str, passengers: int = 1) -> dict:
    """Reserva un vuelo previamente elegido por el usuario.

    Args:
        flight_id: ID del vuelo devuelto por search_flights.
        passenger_name: Nombre completo del pasajero titular.
        passengers: Número de pasajeros.

    Returns:
        dict con status y booking_reference.
    """
    return {
        "status": "success",
        "booking_reference": uuid.uuid4().hex[:6].upper(),
        "flight_id": flight_id,
        "passenger_name": passenger_name,
        "passengers": passengers,
    }


# ---------------------------------------------------------------------------
# Agentes
# ---------------------------------------------------------------------------

weather_agent = Agent(
    name="weather_checker",
    model=MODEL,
    mode="single_turn",  # sin interacción con el usuario
    description="Obtiene el clima actual de una ciudad destino y da un consejo de viaje.",
    instruction=(
        "Recibes el nombre de una ciudad. "
        "1) Usa geocode_address para obtener sus coordenadas. "
        "2) Usa get_weather con esas coordenadas. "
        "3) Usa user_info para conocer las preferencias del usuario. "
        "Devuelve un WeatherReport con un consejo de viaje breve en español."
    ),
    tools=[get_weather, user_info, geocode_address],
    output_schema=WeatherReport,
)

flight_agent = Agent(
    name="flight_booker",
    model=MODEL,
    mode="task",  # puede hacer preguntas al usuario
    description="Busca y reserva vuelos. Puede preguntar al usuario lo que falte.",
    instruction=(
        "Recibes un FlightInput. "
        "1) Llama a search_flights y muestra las opciones al usuario de forma clara. "
        "2) Pregunta cuál vuelo prefiere y el nombre completo del pasajero titular si no lo tienes. "
        "3) Reserva con book_flight solo después de que el usuario elija. "
        "4) Termina con finish_task entregando un FlightResult. "
        "Si el usuario desiste, termina con status='cancelled'."
    ),
    tools=[
        search_flights,
        # HITL: ADK pide confirmación explícita antes de ejecutar la reserva.
        FunctionTool(book_flight, require_confirmation=True),
    ],
    input_schema=FlightInput,
    output_schema=FlightResult,
)

# `adk run` / `adk web` buscan una variable llamada root_agent.
root_agent = Agent(
    name="travel_planner",  # coordinador (modo chat por defecto; no lleva `mode`)
    model=MODEL,
    description="Planificador de viajes que coordina clima y vuelos.",
    instruction=(
        "Eres un planificador de viajes amable que responde en español. "
        "Para preguntas de clima de un destino, delega en weather_checker. "
        "Para buscar o reservar vuelos, delega en flight_booker construyendo un FlightInput "
        "(convierte ciudades a códigos IATA: Lima=LIM, Cusco=CUZ, Arequipa=AQP, Piura=PIU; "
        "fechas en formato YYYY-MM-DD). "
        "Cuando un subagente termine, resume el resultado al usuario."
    ),
    sub_agents=[weather_agent, flight_agent],
    # ADK inyecta tools de delegación con el nombre de cada subagente:
    # weather_checker, flight_booker
)