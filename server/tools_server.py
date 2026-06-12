from fastmcp import FastMCP

mcp = FastMCP("Dummy Tools Server")


# --- weather ---

@mcp.tool
def get_current_temperature(city: str) -> str:
    """Get the current temperature for a city."""
    print(f"[TOOL CALLED] get_current_temperature(city={city!r})")
    return f"It is currently 22°C in {city}."


@mcp.tool
def get_weather_forecast(city: str) -> str:
    """Get the multi-day weather forecast (rain, sun, wind) for a city."""
    print(f"[TOOL CALLED] get_weather_forecast(city={city!r})")
    return f"Forecast for {city}: sunny today, rain expected tomorrow."


# --- calendar ---

@mcp.tool
def create_event(title: str, date: str, time: str) -> str:
    """Create a calendar event with a title, date, and time."""
    print(f"[TOOL CALLED] create_event(title={title!r}, date={date!r}, time={time!r})")
    return f"Created event '{title}' on {date} at {time}."


@mcp.tool
def list_events(date: str) -> str:
    """List all calendar events for a given date."""
    print(f"[TOOL CALLED] list_events(date={date!r})")
    return f"Events on {date}: Team sync at 10:00, Dentist at 15:00."


# --- files ---

@mcp.tool
def read_file(path: str) -> str:
    """Read and return the contents of a file at the given path."""
    print(f"[TOOL CALLED] read_file(path={path!r})")
    return f"Contents of {path}: 'Hello, world!'"


@mcp.tool
def search_files(query: str) -> str:
    """Search for files whose name or contents match a query string."""
    print(f"[TOOL CALLED] search_files(query={query!r})")
    return f"Found 2 files matching '{query}': notes.txt, report.docx"


if __name__ == "__main__":
    mcp.run()
