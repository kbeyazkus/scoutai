import requests
import time

API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
endpoints = {
    "standings_season": "https://api.sportmonks.com/v3/football/standings/seasons/25583?include=participant;rule.type;details.type;form;stage;league;group",
    "fixture_news": "https://api.sportmonks.com/v3/football/fixtures/19427160?include=prematchNews.lines;postmatchNews.lines;participants;league;venue;state;scores;events.type"
}

for name, url in endpoints.items():
    sep = "&" if "?" in url else "?"
    r = requests.get(f"{url}{sep}api_token={API_KEY}")
    print(f"[{r.status_code}] {name}")
    time.sleep(10)
