import requests
import time

API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
tests = {
    "standings": "https://api.sportmonks.com/v3/football/standings/live",
    "news": "https://api.sportmonks.com/v3/football/news",
    "probabilities": "https://api.sportmonks.com/v3/football/predictions/probabilities"
}

for key, url in tests.items():
    r = requests.get(f"{url}?api_token={API_KEY}")
    print(f"[{r.status_code}] {key}")
    time.sleep(10)
