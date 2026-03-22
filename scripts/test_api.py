import requests
API_KEY = "CpJnwdzG4RHQIs37bRQehp2iET5I7fQLoOCUMkT7qxNF78FABDIciJ7ka6o6"
url = f"https://api.sportmonks.com/v3/football/fixtures/19427160?include=news&api_token={API_KEY}"
r = requests.get(url)
print(f"[{r.status_code}] News Final Test")
