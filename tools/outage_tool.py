"""
Mock outage checking tool
"""

def check_outage(area: str) -> str:
    # Simulated API response
    outages = {
        "california": "There is a known outage affecting multiple users.",
        "new york": "No outage detected.",
    }

    return outages.get(area.lower(), "No outage information available.")