from datetime import datetime, timedelta
from typing import Any, Dict, Final

import idna
from dateutil.parser import parse
from fastapi import Depends, Request
from fastapi.exceptions import HTTPException
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param
from jose import jwt
from jose import jwt as _jwt
from jose.constants import ALGORITHMS
from jose.exceptions import JWTClaimsError
from pydantic import BaseModel
from starlette.status import HTTP_401_UNAUTHORIZED

from src.apps.server.admin.schema.auth import SigninOutput
from src.apps.server.exception import JSendError
from src.common.conf.settings import settings
from src.common.utils.crypto import decrypt_db
from src.packages.db.connect import connection


class User(BaseModel):
    idx: int
    status: str
    key: str | None = None


Payload = Dict[str, Any]


SECRET_KEY: Final[str] = getattr(settings, 'SECRET_KEY')
JWT_SECRET_KEY: Final[str] = getattr(settings, 'JWT_SECRET_KEY', SECRET_KEY)
JWT_ALGORITHM: Final[str] = getattr(
    settings, 'JWT_ALGORITHM', ALGORITHMS.HS256)
JWT_ACCESS_TOKEN_VALUE: Final[str] = getattr(
    settings, 'JWT_ACCESS_TOKEN_VALUE', 'access_token')
JWT_REFRESH_TOKEN_VALUE: Final[str] = getattr(
    settings, 'JWT_REFRESH_TOKEN_VALUE', 'refresh_token')


class JWT:
    @staticmethod
    def encode(payload: Payload, is_access_token: bool = False) -> str:
        access_token = JWT_REFRESH_TOKEN_VALUE
        if is_access_token:
            access_token = JWT_ACCESS_TOKEN_VALUE
        token: str = _jwt.encode(
            payload,
            key=JWT_SECRET_KEY,
            algorithm=JWT_ALGORITHM,
            access_token=access_token,
        )

        return token

    @staticmethod
    async def decode(token: str, is_access_token: bool = False, is_admin: bool = False) -> Payload:
        now = datetime.now()
        access_token = JWT_REFRESH_TOKEN_VALUE
        if is_access_token:
            access_token = JWT_ACCESS_TOKEN_VALUE
        try:
            payload: Payload = _jwt.decode(
                token,
                key=JWT_SECRET_KEY,
                algorithms=JWT_ALGORITHM,
                access_token=access_token,
            )
            exp_date: datetime = parse(payload['exp_date'])

            if exp_date < now:
                raise JSendError(
                    code='999999',
                    message="Not authenticated"
                )

            login_status = payload['status']
            idx = payload['idx']

            async with connection() as session:
                if login_status == 'admin':
                    login_info = await session.fetchrow(
                        f"""
                            SELECT
                                idx
                            FROM admin_master
                            WHERE idx = $1
                        """,
                        idx
                    )

                    return {'idx': login_info['idx'], 'status': login_status}

                elif login_status == 'partner':
                    if is_admin:
                        raise JSendError(
                            code='999999', message="Not authenticated",
                        )

                    login_info = await session.fetchrow(
                        f"""
                            SELECT
                                idx,
                                {decrypt_db('site_key')} as site_key
                            FROM partner_master
                            WHERE idx = $1
                        """,
                        idx
                    )
                    return {'idx': login_info['idx'], 'status': login_status, 'key': login_info['site_key']}

                else:
                    login_info = {}

            if not login_info:
                raise JSendError(code='999999', message="Not authenticated")

        except:
            raise JSendError(code='999999', message="Not authenticated")

        return {}


class AuthJWTBearer(HTTPBearer):
    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials:
        authorization : str = ""
        if request is not None:
            authorization: str = request.headers.get('Authorization', "")
        scheme, credentials = get_authorization_scheme_param(authorization)
        if not (authorization and scheme and credentials):
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail='Not authenticated',
                )
            else:
                raise

        if scheme.lower() != 'bearer':
            if self.auto_error:
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail='Invalid authentication credentials',
                )
            else:
                raise

        return HTTPAuthorizationCredentials(scheme=scheme, credentials=credentials)


auth_jwt_schema = AuthJWTBearer()


async def get_user_by_jwt_token(
    token: str,
    is_access_token: bool = False,
    is_admin: bool = False,
) -> User:
    payload: Payload = await JWT.decode(
        token=token,
        is_access_token=is_access_token,
        is_admin=is_admin,
    )

    user_info = User(**payload)
    return user_info


async def auth_jwt_access_user(credentials=Depends(auth_jwt_schema)) -> User:
    token: str = credentials.credentials
    user_info: User = await get_user_by_jwt_token(token=token, is_access_token=True, is_admin=False)

    return user_info


async def auth_jwt_access_admin_user(credentials=Depends(auth_jwt_schema)) -> User:
    token: str = credentials.credentials
    user_info: User = await get_user_by_jwt_token(token=token, is_access_token=True, is_admin=True)

    return user_info

JWTAuthentication = Depends(auth_jwt_access_user)


async def issue_tokens(
    user_idx: int,
    status: str # "partner" | "admin"
):
    now = datetime.now()
    access_token_exp_date = (now + timedelta(days=7)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    refresh_token_exp_date = (now + timedelta(days=14)).strftime(
        "%Y-%m-%d 00:00:00"
    )
    # jwt = JWT()
    access_token = jwt.encode(
        {
            "idx": user_idx,
            "exp_date": str(access_token_exp_date),
            "status": status,
        },
        settings.SECRET_KEY,
        algorithm="HS256",
    )
    refresh_token = jwt.encode(
        {
            "idx": user_idx,
            "exp_date": str(refresh_token_exp_date),
            "status": status,
        },
        settings.SECRET_KEY,
        algorithm="HS256",
    )

    # return {
    #     "idx": user_idx,
    #     "message": "로그인에 성공했습니다.",
    #     "access_token": access_token,
    #     "refresh_token": refresh_token,
    # }
    return SigninOutput(
        idx=user_idx,
        message="로그인에 성공했습니다.",
        access_token=access_token,
        refresh_token=refresh_token,
    )


# TODO: order-site-payment-validate-api 에서 사용되던 함수 -> 핫파에서 테스트 필요
def decode_punycode(url):
    try:
        # decoded_url = decode(url).encode('utf-8').decode('utf-8')
        decoded_url = idna.decode(url)
        return decoded_url
    except Exception as e:
        print(f"Error decoding URL: {e}")
        return url
