import requests
API_KEY = "BURAYA_API_ANAHTARINI_YAZ"
url = f"https://api.sportmonks.com/v3/football/fixtures/19427160?include=news&api_token={API_KEY}"
r = requests.get(url)
print(f"[{r.status_code}] News Final Test")
