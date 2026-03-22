import requests

API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
FIXTURE_ID = "19622007"
# 27 başlığın teknik karşılıkları
INCLUDES = [
    "participants", "scores", "events", "lineups", "odds", "statistics",
    "predictions", "trends", "pressure", "standings", "topscorers",
    "tvstations", "sidelined", "referees", "comments", "venue", "coaches",
    "periods", "league", "season", "form", "groupstandings", "news", 
    "metadata", "state", "weatherReport", "probabilities"
]

for inc in INCLUDES:
    url = f"https://api.sportmonks.com/v3/football/fixtures/{FIXTURE_ID}?api_token={API_KEY}&include={inc}"
    try:
        r = requests.get(url, timeout=10)
        print(f"[{r.status_code}] {inc}")
    except:
        print(f"[HATA] {inc}")
