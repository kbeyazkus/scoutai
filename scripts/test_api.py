import requests
import time

API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
failed_urls = [
    "https://api.sportmonks.com/v3/football/fixtures/19427160?include=participants;league;venue;state;scores;events.type;events.period;events.player;xGFixture.type;lineups.player;lineups.xGlineup.type;lineups.details.type",
    "https://api.sportmonks.com/v3/football/players/154421?include=latest.xGlineup.type;statistics.details.type;statistics.season.league;statistics.position;latest.fixture.participants;latest.fixture.scores;nationality;latest.fixture.events;latest.fixture.league;teams.team&filters=playerstatisticSeasons:25583;playerstatisticdetailTypes:5304",
    "https://api.sportmonks.com/v3/football/fixtures/19427160?include=participants;league;venue;state;scores;events.type;events.player;pressure.participant",
    "https://api.sportmonks.com/v3/football/fixtures/19427160?include=prematchNews.lines;postmatchNews.lines;participants;league;venue;state;scores;events.type",
    "https://api.sportmonks.com/v3/football/teams/9?include=latest.statistics.type;latest.xgfixture.type;latest.participants;latest.scores.type"
]

for url in failed_urls:
    sep = "&" if "?" in url else "?"
    r = requests.get(f"{url}{sep}api_token={API_KEY}")
    print(f"[{r.status_code}] {url.split('football/')[1][:40]}...")
    time.sleep(20)
