import json
from http import HTTPStatus
from typing import Any, Dict, Tuple

import aiohttp

# import urllib.parse


async def call_api(
    domain: str,
    method: str,
    resource_path: str  = '',
    timeout=4,
    retry_count=1,
    **kwargs
) :
    """
    api 콜 관련 메서드
    """
    try:
        for try_count in range(retry_count):
            async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(verify_ssl=False)
            ) as session:
                _request = getattr(session, method.lower(), None)
                if not _request:
                    raise Exception(f"Not supported http method. ({method})")
                url = domain
                if url[-1] == '/':
                    url = url[:-1]
                if resource_path:
                    url += f'/{resource_path}'
                response = await _request(
                    url=url,
                    **kwargs,
                    timeout=timeout,
                )
                response_text = await response.text()
                if response.status != HTTPStatus.OK and try_count + 1 < retry_count:
                    continue
                try:
                    return response.status, json.loads(response_text)
                except:
                    return response.status, response_text
    except TimeoutError as e:
        return HTTPStatus.REQUEST_TIMEOUT, str(e)
    except Exception as e:
        return HTTPStatus.INTERNAL_SERVER_ERROR, str(e)

    # 기본 반환값 추가

# async def call_api(
#     domain: str,
#     method: str,
#     resource_path: str | None = '',
#     timeout=4,
#     retry_count=1,
#     **kwargs
# ) -> Tuple[int, Dict[str, Any]]:
#     """
#     api 콜 관련 메서드
#     """
#     try:
#         for try_count in range(retry_count):
#             async with aiohttp.ClientSession(
#                 connector=aiohttp.TCPConnector(verify_ssl=False)
#             ) as session:
#                 _request = getattr(session, method.lower(), None)
#                 if not _request:
#                     raise Exception(f"Not supported http method. ({method})")

#                 url = domain
#                 if url[-1] == '/':
#                     url = url[:-1]
#                 if resource_path:
#                     url += f'/{resource_path}'

#                 # Construct the full URL with query string if data is provided
#                 if 'data' in kwargs:
#                     query_string = urllib.parse.urlencode(kwargs['data'])
#                     full_url = f"{url}?{query_string}"
#                 else:
#                     full_url = url

#                 # Log the full URL and request data
#                 print(f"Full URL: {full_url}")
#                 if 'json' in kwargs:
#                     print(f"Request JSON: {json.dumps(kwargs['json'])}")
#                 elif 'data' in kwargs:
#                     print(f"Request Data: {kwargs['data']}")

#                 response = await _request(
#                     url=url,
#                     **kwargs,
#                     timeout=timeout,
#                 )
#                 response_text = await response.text()
#                 if response.status != HTTPStatus.OK and try_count + 1 < retry_count:
#                     continue
#                 try:
#                     return response.status, json.loads(response_text)
#                 except:
#                     return response.status, response_text
#     except TimeoutError as e:
#         return HTTPStatus.REQUEST_TIMEOUT, str(e)
#     except Exception as e:
#         return HTTPStatus.INTERNAL_SERVER_ERROR, str(e)
