import os
import time
import uuid
import requests


class GigaChatClient:
    """
    Работает в 2 режимах:
    1) Если задан GIGACHAT_TOKEN (access token) — используем его.
    2) Иначе если задан GIGACHAT_AUTH_KEY (Authorization key) — автоматически получаем access token
       через POST https://ngw.devices.sberbank.ru:9443/api/v2/oauth и кешируем ~30 минут.
    """

    def __init__(self, api_base: str):
        self.api_base = (api_base or "").rstrip("/")
        self.model = os.environ.get("GIGACHAT_MODEL", "GigaChat")
        self.scope = os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

        v = (os.environ.get("GIGACHAT_VERIFY_SSL", "true") or "").strip().lower()
        self.verify_ssl = not (v in {"0", "false", "no"})

        self.oauth_url = os.environ.get("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")

        self._cached_token = None
        self._expires_at = 0.0

    def _get_env_access_token(self) -> str:
        return (os.environ.get("GIGACHAT_TOKEN") or "").strip()

    def _get_auth_key(self) -> str:
        return (os.environ.get("GIGACHAT_AUTH_KEY") or "").strip()

    def _get_access_token_via_oauth(self) -> str:
        # кеш
        now = time.time()
        if self._cached_token and now < self._expires_at:
            return self._cached_token

        auth_key = self._get_auth_key()
        if not auth_key:
            return ""

        # key иногда уже приходит с "Basic ..."
        if auth_key.lower().startswith("basic "):
            auth_header = auth_key
        else:
            auth_header = f"Basic {auth_key}"

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": auth_header,
        }
        data = {"scope": self.scope}

        r = requests.post(self.oauth_url, headers=headers, data=data, timeout=30, verify=self.verify_ssl)
        if r.status_code != 200:
            return ""

        try:
            j = r.json()
        except Exception:
            return ""

        token = (j.get("access_token") or j.get("accessToken") or "").strip()
        if not token:
            return ""

        # expires_in обычно в секундах (примерно 1800)
        expires_in = j.get("expires_in") or j.get("expiresIn")
        try:
            expires_in = int(expires_in) if expires_in is not None else 1800
        except Exception:
            expires_in = 1800

        # ставим небольшой запас, чтобы не словить протухание
        self._cached_token = token
        self._expires_at = time.time() + max(60, expires_in - 60)
        return token

    def _get_token(self) -> str:
        # приоритет: явно заданный access token
        env_token = self._get_env_access_token()
        if env_token:
            return env_token

        # иначе пробуем получить через Authorization key
        return self._get_access_token_via_oauth()

    def ask(self, prompt: str) -> str:
        token = self._get_token()
        if not token:
            return ""

        url = f"{self.api_base}/chat/completions"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }

        r = requests.post(url, json=payload, headers=headers, timeout=60, verify=self.verify_ssl)

        # если внезапно 401/403 и у нас есть AUTH_KEY — возможно протух env token.
        if r.status_code in (401, 403) and self._get_auth_key():
            self._cached_token = None
            self._expires_at = 0.0
            token2 = self._get_access_token_via_oauth()
            if not token2:
                return ""
            headers["Authorization"] = f"Bearer {token2}"
            r = requests.post(url, json=payload, headers=headers, timeout=60, verify=self.verify_ssl)

        if r.status_code != 200:
            return ""

        try:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            return ""
