import requests
import time

API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
# 404 ve 403 verenleri ayirarak tekrar test et
urls = [
    "https://api.sportmonks.com/v3/football/standings/seasons/25583?include=participant;rule.type;details.type;form;stage;league;group",
    "https://api.sportmonks.com/v3/football/fixtures/19427160?include=participants;league;venue;state;scores;events.type"
]

for url in urls:
    final_url = f"{url}&api_token={API_KEY}"
    r = requests.get(final_url)
    print(f"[{r.status_code}] Sorgu: {url[:50]}...")
    if r.status_code != 200:
        print(f"Hata Detayi: {r.text}")
    time.sleep(10)
