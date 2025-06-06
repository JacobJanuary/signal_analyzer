"""Base API client module."""
import time
from typing import Dict, Any, Optional, Union
import requests
from utils.logger import setup_logger, log_with_context
from config.settings import settings


logger = setup_logger(__name__)


class BaseAPIClient:
    """Base API client with retry logic."""

    def __init__(self, base_url: str):
        """Initialize base API client."""
        self.base_url = base_url
        self.session = requests.Session()

    def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        method: str = 'GET'
    ) -> Optional[Union[Dict, list]]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"
        params = params or {}

        for attempt in range(settings.RETRY_ATTEMPTS):
            try:
                response = self.session.request(method, url, params=params)
                response.raise_for_status()
                return response.json()

            except requests.exceptions.HTTPError as e:
                error_details = {}
                if e.response:
                    error_details['status_code'] = e.response.status_code
                    try:
                        error_details['response_json'] = e.response.json()
                    except:
                        error_details['response_text'] = e.response.text[:200]  # First 200 chars

                log_with_context(
                    logger, 'error',
                    f"HTTP error on attempt {attempt + 1}",
                    url=url,
                    params=params,
                    **error_details
                )
                if attempt < settings.RETRY_ATTEMPTS - 1:
                    time.sleep(settings.RETRY_DELAY)
                else:
                    return None

            except requests.exceptions.RequestException as e:
                log_with_context(
                    logger, 'error',
                    f"Request error on attempt {attempt + 1}",
                    url=url,
                    error=str(e)
                )
                if attempt < settings.RETRY_ATTEMPTS - 1:
                    time.sleep(settings.RETRY_DELAY)
                else:
                    return None

            except ValueError as e:
                log_with_context(
                    logger, 'error',
                    "JSON decode error",
                    url=url,
                    error=str(e)
                )
                return None

        return None