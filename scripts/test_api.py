import requests

API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
FIXTURE_ID = "19622007"
INCLUDES = [
    "participants", "league.country", "venue", "state", "scores", 
    "periods", "events.type", "events.period", "events.player", 
    "statistics.type", "lineups.player", "lineups.type", 
    "lineups.details.type", "metadata.type", "coaches", 
    "sidelined.sideline.player", "sidelined.sideline.type", 
    "weatherReport", "comments", "pressure.participant", 
    "trends.type", "trends.participant"
]

for inc in INCLUDES:
    url = f"https://api.sportmonks.com/v3/football/fixtures/{FIXTURE_ID}?api_token={API_KEY}&include={inc}"
    try:
        r = requests.get(url, timeout=10)
        print(f"[{r.status_code}] {inc}")
    except Exception as e:
        print(f"[HATA] {inc}")
