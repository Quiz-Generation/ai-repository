import asyncio
import json
import re
from datetime import datetime
from json.decoder import JSONDecodeError
from typing import Any, Dict, Final, List, Optional, TypedDict

from aiohttp import ClientResponseError, ClientSession, TCPConnector
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.apps.core.repository.slack_message import *
from src.apps.server.exception import ErrorCode
from src.common.conf.settings import settings
from src.common.constants import Constants
from src.common.utils.logger import set_logger
from src.packages.slack import AlarmLevel
from src.packages.slack.client import send_alarm_to_thread

LOGGER = set_logger("scheduler.slack")


SLACK_SERVICE_ERROR_WEBHOOK_URI: Final[str] = settings.SLACK_SERVICE_ERROR_WEBHOOK_URI
SLACK_AUTO_CHARGE_WEBHOOK_URI: Final[str] = settings.SLACK_AUTO_CHARGE_WEBHOOK_URI
SLACK_ORDER_WEBHOOK_URI: Final[str] = settings.SLACK_ORDER_WEBHOOK_URI
SLACK_ORDER_SITE_BALANCE_WEBHOOK_URI: Final[str] = settings.SLACK_ORDER_SITE_BALANCE_WEBHOOK_URI
AISTAGRAM_SLACK_BOT_TOKEN: Final[str] = settings.AISTAGRAM_SLACK_BOT_TOKEN
SLACK_ORDER_SITE_BALANCE_CHANNEL_ID: Final[str] = settings.SLACK_ORDER_SITE_BALANCE_CHANNEL_ID



def is_working_hours():
        now = datetime.now()
        # 평일(월-금)인지 확인
        if now.weekday() < 5:
            # 아침 8시 ~ 저녁 8시 사이인지 확인
            if 8 <= now.hour < 20:
                return True
        return False


async def get_alimtalk_error_message(
    error_code
):
    for attr_name in dir(ErrorCode.Alimtalk):
        # 클래스의 내부 메서드/속성 제외
        if not attr_name.startswith('__'):
            error_tuple = getattr(ErrorCode.Alimtalk, attr_name)
            # 주어진 error_code와 일치하는지 확인
            if error_tuple[0] == error_code:
                return error_tuple[1]
    return None  # 일치하는 코드가 없으면 None 반환


async def fix_surrogates(text):
    """
    써로게이트 페어를 단일 유니코드 문자로 변환하는 함수.
    """
    result = []
    i = 0
    while i < len(text):
        char = text[i]
        # High surrogate 감지
        if 0xD800 <= ord(char) <= 0xDBFF and i + 1 < len(text):
            next_char = text[i + 1]
            # Low surrogate 감지 및 변환
            if 0xDC00 <= ord(next_char) <= 0xDFFF:
                codepoint = ((ord(char) - 0xD800) << 10) + (ord(next_char) - 0xDC00) + 0x10000
                result.append(chr(codepoint))
                i += 2
                continue
        result.append(char)
        i += 1
    return ''.join(result)


class SlackBase:
    def __init__(self, token: str | None = None, default_channel: str | None = None):
        self.client = WebClient(token=token) if token else None
        self.default_channel = default_channel



class Button(TypedDict):
    title: str
    url: str


class Logging:
    DEBUG: str = ':white_circle:'
    INFO: str = ':large_green_circle:'
    WARNING: str = ':large_yellow_circle:'
    ERROR: str = ':large_orange_circle:'
    CRITICAL: str = ':red_circle:'

    TARGET_FORMAT = 'format'
    TAX_BILL_INFO: str = ":page_facing_up:"  # 세금 계산서
    CASH_RECEIPTS_INFO: str = ":coin:"  # 현금 영수증
    PANEL_INFO: str = ":globe_with_meridians:"  # 해외 패널
    RING_RING: str = ":ring_ring:" # 관리자 위험 알림
    WARNING_SIGN: str = "warning:" # 관리자 경고 알림

    async def send(self, data: Any, **kwargs):
        pass

    def f(self, title: str, description: str, **kwargs):
        format_name: Optional[str] = kwargs.get('format')
        if format_name is not None:
            func = getattr(self, f'{format_name}_{self.TARGET_FORMAT}', None)
            if func is not None:
                return func(title, description, **kwargs)
        return getattr(self, self.TARGET_FORMAT)(title, description, **kwargs)

    async def debug(self, title: str, description: str, **kwargs):
        await self.send(
            self.f(
                f'{self.DEBUG} [{settings.STAGE}] - {title}',
                description,
                **kwargs,
            ),
            **kwargs,
        )

    async def info(self, title: str, description: str, **kwargs):
        await self.send(
            self.f(
                f"{kwargs.get('emoji_type', self.INFO)} [{settings.STAGE}] - {title}",
                description,
                **kwargs,
            ),
            **kwargs,
        )

    async def warning(self, title: str, description: str, **kwargs):
        await self.send(
            self.f(
                f'{self.WARNING} [{settings.STAGE}] {title}',
                description,
                **kwargs,
            ),
            **kwargs,
        )

    async def error(self, title: str, description: str, **kwargs):
        await self.send(
            self.f(
                f'{self.ERROR} [{settings.STAGE}] {title}',
                description,
                **kwargs,
            ),
            **kwargs,
        )

    async def critical(self, title: str, description: str, **kwargs):
        await self.send(
            self.f(
                f'{self.CRITICAL} [{settings.STAGE}] {title}',
                description,
                **kwargs,
            ),
            **kwargs,
        )


class SlackLogging(SlackBase, Logging):
    async def send(self, data: Dict, **kwargs):
        blocks = data.get('blocks', [])
        total_text_length = sum(len(block['text']['text']) for block in blocks if 'text' in block and 'text' in block['text'])

        if total_text_length > 2500:
            # 블록의 텍스트 길이가 2500자를 초과하는 경우 파일로 업로드
            if self.client:
                try:
                    channel = SLACK_ORDER_SITE_BALANCE_CHANNEL_ID
                    content = "\n".join(block['text']['text'] for block in blocks if 'text' in block and 'text' in block['text'])

                    # 멘션 ID 추출 및 멘션 텍스트 생성
                    mention_ids = re.findall(r'<@(\w+)>', content)
                    title = data.get('text', '').strip()
                    initial_comment = ''
                    if mention_ids:
                        mentions = ' '.join([f'<@{mention_id}>' for mention_id in mention_ids])
                        initial_comment = f"{mentions}\n{title}"

                    # content에서 멘션과 타이틀 제거
                    content = re.sub(r'<@\w+>', '', content).strip()  # 멘션 제거
                    content = content.replace(title, '', 1).strip()  # 타이틀 제거 (첫 번째 발견된 것만)

                    response = self.client.files_upload_v2(
                        channel=channel,
                        content=content,
                        filename="long_message.txt",
                        title=title,
                        initial_comment=initial_comment
                    )
                    return response
                except SlackApiError as e:
                    LOGGER.error(f"Slack SDK 파일 업로드 실패: {e.response['error']}")
                    raise
            else:
                LOGGER.error("Slack 클라이언트가 초기화되지 않았습니다.")
                raise ValueError("Slack 클라이언트가 초기화되지 않았습니다.")
        else:
            # 기존 webhook 방식 유지
            try:
                async with ClientSession(connector=TCPConnector(verify_ssl=False)) as _session:
                    res = await _session.post(
                        url=kwargs['slack_url'],
                        json=data,
                        headers={'Content-Type': 'application/json'}
                    )
                    res.raise_for_status()
                    if res.status != 200:
                        LOGGER.error(await res.text())
            except ClientResponseError as e:
                # try:
                #     LOGGER.error(await e.response.json())
                # except JSONDecodeError:
                #     LOGGER.error(await e.response.text())
                LOGGER.error(f"Error: {e.message}, Status: {e.status}")

    def format(
        self,
        title: str,
        description: str,
        **kwargs,
    ):
        if kwargs.get('tag_type') in [
            'hotpartners-api',
            'hotpartners-scheduler',
            'hotpartners-scheduler-panel-abnormal',
        ]:
            supervisor_slack_unique_ids = [
                'U07A0D6B6H5',  # NOTE: 박준용 사원
                # 'U0239QJ5CFK',  # NOTE: 권순민 실장
                'U02EC6X4WG4',  # NOTE: 김정환 팀장
            ]
        elif kwargs.get('tag_type') in ['parsing']:
            supervisor_slack_unique_ids = [
                'U07A0D6B6H5',  # NOTE: 박준용 사원
                'U02EC6X4WG4',  # NOTE: 김정환 팀장
            ]
        elif kwargs.get('tag_type') in ['decago']:
            supervisor_slack_unique_ids = [
                'U07A8A7NS3G',  # NOTE: 이주현 사원
                'U02EC6X4WG4',  # NOTE: 김정환 팀장
            ]
        elif kwargs.get('tag_type') in ['none', 'autocharge']:
            supervisor_slack_unique_ids = []

        elif kwargs.get('tag_type') in [
            'hotpartners-scheduler-panel-service-change',
        ]:
            if is_working_hours():
                supervisor_slack_unique_ids = [
                    'U04JBRN0M3K',  # NOTE: 송호진 주임
                    'U07A0D6B6H5',  # NOTE: 박준용 사원
                ]
            else:
                supervisor_slack_unique_ids = []

        elif kwargs.get('tag_type') in [
            'developers',
        ]:
            supervisor_slack_unique_ids = [
                'U02EC6X4WG4',  # NOTE: 김정환 팀장님
                'U07A0D6B6H5',  # NOTE: 박준용 사원
            ]

        else:
            supervisor_slack_unique_ids = [
                'U07A0D6B6H5',  # NOTE: 박준용 사원
                # 'U0239QJ5CFK',  # NOTE: 권순민 실장님
                'U02EC6X4WG4',  # NOTE: 김정환 팀장님
            ]

        if kwargs.get('description_plus_1'):
            block_header: List[Dict] = [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': title,
                        'emoji': True,
                    },
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{description}```',
                    },
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{kwargs.get("description_plus_1")}```',
                    },
                },
            ]

        if kwargs.get('description_plus_1') and kwargs.get('description_plus_2'):
            block_header: List[Dict] = [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': title,
                        'emoji': True,
                    },
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{description}```',
                    },
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{kwargs.get("description_plus_1")}```',
                    },
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{kwargs.get("description_plus_2")}```',
                    },
                },
            ]

        elif kwargs.get('button_list', []):
            button_list = kwargs.get('button_list', [])

            block_header: List[Dict] = [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': title,
                        'emoji': True,
                    },
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{description}```',
                    },
                },
                *button_list
            ]
        else:
            block_header: List[Dict] = [
                {
                    'type': 'header',
                    'text': {
                        'type': 'plain_text',
                        'text': title,
                        'emoji': True,
                    },
                },
                {
                    'type': 'section',
                    'text': {
                        'type': 'mrkdwn',
                        'text': f'```{description}```',
                    },
                },
            ]

        if supervisor_slack_unique_ids:
            block_header.insert(1, {
                'type': 'section',
                'text': {
                    'type': 'mrkdwn',
                    'text': (
                        ', '.join(
                            [f'<@{_id}>' for _id in supervisor_slack_unique_ids]
                        )
                    ),
                },
            })

        block_content: List[Dict] = kwargs.pop('content', [])
        block_footer: List[Dict] = []

        return {
            'text': title,
            'blocks': block_header + block_content + block_footer,
        }


class SlackBotLogging(SlackLogging):
    def __init__(self, token: str = AISTAGRAM_SLACK_BOT_TOKEN, default_channel: str | None = None):
        super().__init__(token=token, default_channel=default_channel)

    def error_consecutive_fail_format(
        self,
        title: str,
        description: str,
        **kwargs,
    ):
        buttons_data = kwargs.pop("button", [])

        # buttons_data가 리스트가 아니면 빈 리스트로 대체
        if not isinstance(buttons_data, list):
            buttons_data = []

        elements = [
            {
                "type": "button",
                "text": {
                    "type": "plain_text",
                    "text": data.get("title", "Default Error Title"),
                    "emoji": True,
                },
                "value": "button_click_event",
                "url": data.get("url", "http://default_url.com"),  # URL이 없거나 유효하지 않은 경우 기본 URL 사용
                "action_id": "button-action",
            }
            for data in buttons_data if 'title' in data and 'url' in data  # title과 url 키가 모두 있는 항목에 대해서만 버튼 생성
        ]

        button_section = {
            "type": "actions",
            "elements": elements,
        } if elements else {}  # elements가 비어 있지 않은 경우에만 button_section 생성

        # content에 button_section이 유효한 경우에만 포함
        return self.format(title, description, content=[button_section] if button_section else [], **kwargs)


    # API 비정상 동작 알림
    async def error_service_api_abnormal(self, err_obj_name: str) -> None:
        await self.critical(
            f"핫파트너스 - API 비정상 동작 알림",
            f"에러 메시지: {err_obj_name}",
            tag_type='hotpartners-api',
            slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    # 스케줄러 비정상 동작 알림
    async def error_service_scheduler_abnormal(self, err_obj_name: str, caller: str = "") -> None:
        await self.critical(
            f"핫파트너스 - {caller} 스케줄러 비정상 동작 알림",
            f"에러 메시지: {err_obj_name}",
            tag_type='hotpartners-scheduler',
            slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    # DB 풀의 가용 커넥션이 낮아질 때 슬랙 알림
    async def db_connection_capacity_in_danger(self, err_obj_name: str, caller: str = "") -> None:
        await self.critical(
            f"핫파트너스 - {caller} DB Pool 가용 커넥션 부족 경고",
            f"에러 메시지: {err_obj_name}",
            tag_type='hotpartners-scheduler',
            slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    # DB connection 비정상 동작 알림
    async def error_scheduler_db_connection_abnormal(self, err_obj_name: str) -> None:
        await self.critical(
            f"핫파트너스 - 스케줄러 DB 비정상 동작 알림",
            f"에러 메시지: {err_obj_name}",
            tag_type='hotpartners-scheduler',
            slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    # 알림톡 api 응답 에러 알림
    async def error_order_service_alimtalk_abnormal(
        self,
        name: str,
        phone_number: int,
        alimtalk_type: str,
        err_msg: str,
        err_code: str,
        **kwargs,
    ) -> None:
        order_stage = settings.ORDER_STAGE
        await self.error(
            title=f"핫파트너스 - 포인트 알림톡 API 응답 에러 알림",
            description=(
                f"유형: {alimtalk_type}\n"
                f"파트너 이름: {name}\n"
                f"파트너 전화번호: {phone_number}\n"
                f"에러 코드: {err_code}\n"
                f"에러 메시지: {err_msg}\n"
            ),
            tag_type='hotpartners-api',
            slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    async def error_order_service_alimtalk_token_abnormal(
        self,
        alimtalk_type,
        err_msg,
        err_code,
    ) -> None:
        await self.error(
            title=f"핫파트너스 - 포인트 알림톡 API 응답 에러 알림",
            description=(
                f"유형: {alimtalk_type}\n"
                f"에러 코드: {err_code}\n"
                f"에러 메시지: {err_msg}\n"
            ),
            tag_type='hotpartners-api',
            slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    async def error_order_service_alimtalk_response_abnormal(self, **kwargs) -> None:
        if kwargs["status"] == "traceback":
            await self.error(
                title=f"핫파트너스 - 포인트 알림톡 전송 결과 API 비정상 동작 알림",
                description=(
                    f"파트너 전화번호: {kwargs['phone_number']}\n"
                    f"전송 타입: {kwargs['media']}\n"
                    f"결과 코드: {kwargs['code']}\n"
                    f"에러 메시지: {kwargs['err_msg']}\n"
                    f"MSGID: {kwargs['msgid']}\n"
                ),
                tag_type='hotpartners-api',
                slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
            )
        else:
            if kwargs['code']:
                error_message = await get_alimtalk_error_message(
                    error_code=kwargs['code']
                )
            await self.error(
                title=f"핫파트너스 - 포인트 알림톡 전송 결과 실패 알림",
                description=(
                    f"파트너 이름: {kwargs['site_name']} ({kwargs['site_idx']})\n"
                    f"파트너 전화번호: {kwargs['phone_number']}\n"
                    f"전송 타입: {kwargs['media']}\n"
                    f"에러 코드: {kwargs['code']}\n"
                    f"에러 메세지: {error_message}\n"
                    f"전송 시간: {kwargs['trans_time']}\n"
                    f"MSGID: {kwargs['msgid']}\n"
                ),
                tag_type='hotpartners-api',
                # slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI, # 임시 오류 알림방
                slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI,
            )

    async def send_tax_bill_info(self, **kwargs) -> None:  # 세금 계산서
        await self.info(
            title=f"세금계산서 발급요청 ({kwargs['deposit_at']})",
            description=(
                f"사업자번호: {kwargs['tax_business']}\n"
                f"회 사 명: {kwargs['tax_company']}\n"
                f"대표자명: {kwargs['tax_ceo']}\n"
                f"이메일주소: {kwargs['tax_email']}\n"
                f"충전 금액: {kwargs['charged_amount']} (vat 포함)\n"
            ),
            emoji_type=self.TAX_BILL_INFO,
            tag_type='autocharge',
            slack_url=SLACK_AUTO_CHARGE_WEBHOOK_URI,
        )

    async def send_cash_receipts_info(self, **kwargs) -> None:  # 현금 영수증
        try:
            phone_number = '-'.join(
                [
                    kwargs['receipt_phone'][:3],
                    kwargs['receipt_phone'][3:7],
                    kwargs['receipt_phone'][7:]
                ],
            )
        except:
            phone_number = kwargs['receipt_phone']

        await self.info(
            title=f"현금영수증 발급요청 ({kwargs['deposit_at']})",
            description=(
                f"충전자명: {kwargs['charge_depositor']}\n"
                f"전화번호: {phone_number}\n"
                f"충전 금액: {kwargs['charged_amount']} (vat 포함)\n"
            ),
            emoji_type=self.CASH_RECEIPTS_INFO,
            tag_type='autocharge',
            slack_url=SLACK_AUTO_CHARGE_WEBHOOK_URI,
        )

    async def send_slack_alert_comment_generation_failed(
        self,
        order_idx: int,
        err_msg: str,
        **kwargs,
    ):
        # TODO:
        pass


    async def send_slack_alert_orders_parse_failed(
        self,
        order_idx_list: list,
        err_msg: str,
        **kwargs,
    ):
        await self.warning(
            title=f"핫파트너스 - {err_msg} 알림",
            description=(
                f"주문 idx: {list(order_idx_list)}\n"
            ),
            format="error_consecutive_fail",
            tag_type='hotpartners-scheduler',
            slack_url=SLACK_ORDER_WEBHOOK_URI,
        )


    # 주문 선검증 실패용 Cancel 알림
    async def send_slack_alert_order_pre_check_failed(
        self,
        order_idx: int,
        err_msg: str,
        **kwargs,
    ):
        request_data_obj = await get_order_request_data(
            order_idx=order_idx
        )
        if not request_data_obj:
            return # NOTE: 임시방편

        if request_data_obj:
            request_data = json.loads(request_data_obj['request_data'])

            partner_idx = request_data_obj['partner_idx']
            site_name = request_data_obj['site_name']
            service_name = request_data_obj['service_name']
            service_number = request_data.pop('service_number', None)
            order_count = request_data.pop('order_count', None)
            link = request_data.pop('link', None)

        await self.warning(
            title=f"핫파트너스 - 스케줄러 주문 {Constants.Order.Status.CANCELED} 알림",
            description=(
                f"파트너 이름: {site_name} ({partner_idx})\n"
                f"주문 서비스: {service_name} ({service_number})\n"
                f"주문 수량: {order_count}\n"
                f"주문 idx: {order_idx}\n"
                f"url: {link}\n"
                f"에러 메세지: {err_msg}\n"
            ),
            format="error_consecutive_fail",
            tag_type='none',
            slack_url=SLACK_ORDER_WEBHOOK_URI,
        )


    # 주문 Cancel | Partial 알림
    async def send_slack_alert_order_status(
        self,
        order_idx: int,
        err_msg: str,
        status: str,
        **kwargs,
    ):
        err_msg = await fix_surrogates(err_msg)
        order_detail_info = await get_order_detail_info(order_idx=order_idx)
        if status == Constants.Order.Status.CANCELED:
            status = 'Cancel' # NOTE: 확실한 실패 알림과 타이틀 통일

        button_url = order_detail_info.get('link', '')  # None이 아닌 빈 문자열로 기본값 설정
        is_valid_url = isinstance(button_url, str) and button_url.startswith(('http://', 'https://'))

        await self.warning(
            title=f"핫파트너스 - 스케줄러 주문 {status} 알림",
            description=(
                f"파트너 이름: {order_detail_info['site_name']} ({order_detail_info['partner_idx']})\n"
                f"주문 서비스: {order_detail_info['service_name']} ({order_detail_info['service_number']})\n"
                f"주문 수량: {order_detail_info['order_count']}\n"
                f"주문 idx: {order_idx}\n"
                f"url: {order_detail_info['link']}\n"
                f"에러 메세지: {err_msg}\n"
            ),
            # button=[Button(title="바로가기", url=order_detail_info['link'])],
            button=[Button(title="바로가기", url=button_url)] if is_valid_url else None,  # 조건부 버튼 추가
            format="error_consecutive_fail",
            tag_type='none',
            slack_url=SLACK_ORDER_WEBHOOK_URI,
        )

    # 패널 서비스 정보 변경 알림 (e.g. 단가 / 최대수량 / 최소수량) -> 안내
    async def send_slack_for_panel_service_change_info(
        self,
        activate_logs,
        deactivate_logs,
        update_logs,
        **kwargs,
    ) -> None:
        await self.info(
            title=f"핫파트너스 - 스케줄러 패널 서비스 변경 알림",
            description=(
                "패널 서비스 활성화 정보\n\n"
                f"{''.join(activate_logs)}"
            ),
            description_plus_1=(
                "패널 서비스 비활성화 정보\n\n"
                f"{''.join(deactivate_logs)}"
            ),
            description_plus_2=(
                "패널 서비스 정보 변경\n\n"
                f"{''.join(update_logs)}"
            ),
            tag_type='hotpartners-scheduler-panel-service-change',
            slack_url=settings.SLACK_ORDER_SITE_BALANCE_WEBHOOK_URI, # 셀프마케팅플랫폼-잔고-알림방
            # slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI, # 임시-핫파트너스-오류-알림방
        )

    # 패널 서비스 정보 변경 알림 (e.g. 단가 / 최대수량 / 최소수량) -> 경고
    async def send_slack_for_panel_service_change_warning(
        self,
        change_warning_logs,
        **kwargs,
    ) -> None:
        try:
            LOGGER.info(f"change_warning_logs: {change_warning_logs}")
            mentions = []
            if is_working_hours():
                mentions=[
                    "U04JBRN0M3K", # 송호진 주임
                    "U07A0D6B6H5", # 박준용 사원
                ]

            task = asyncio.create_task(
                send_alarm_to_thread(
                    logger=LOGGER,
                    level=AlarmLevel.WARNING,
                    title=f"핫파트너스 - 스케줄러 패널 서비스 변경 알림",
                    channel_id=settings.SLACK_ORDER_SITE_BALANCE_CHANNEL_ID,  # 셀프마케팅플랫폼-잔고-알림방
                    messages=[f"{''.join(change_warning_logs)}"],
                    extra_blocks=[],
                    mentions=mentions,
                    try_cnt=1,
                )
            )

            await task

            if task.exception():
                raise task.exception()

        except Exception as e:
            LOGGER.error(
                f"""
                    슬랙 메세지 전송 실패:
                    error: {str(e)}
                """
            )


    # 패널 서비스 조회 불가할 때 슬랙 알람 (e.g. 패널 내부 서버 에러일 경우)
    async def send_slack_for_panel_unavailability(
        self,
        request_failed_logs,
        # err_msg: str = "패널 서비스 비정상 응답 감지",
        **kwargs,
    ) -> None:
        await self.warning(
            title=f"핫파트너스 - 스케줄러 패널 서비스 비정상 응답 감지 알림",
            description=(
                "서비스 조회 실패한 패널명\n\n"
                f"패널명: {', '.join(request_failed_logs)}"
            ),
            tag_type='hotpartners-scheduler-panel-abnormal',
            emoji_type=self.PANEL_INFO,
            slack_url=settings.SLACK_ORDER_SITE_BALANCE_WEBHOOK_URI, # 셀프마케팅플랫폼-잔고-알림방
            # slack_url=settings.SLACK_SERVICE_ERROR_WEBHOOK_URI, # 임시-핫파트너스-오류-알림방
        )

    # 무한 주문 방지 테이블 슬랙 알람
    async def send_slack_alert_for_temp_running_order_detail_list(
        self,
        temp_running_idx_list: Any,
        err_msg: str = "무한 주문 방지. 방지된 목록 감지 알림"
    ) -> None:
        await self.warning(
            title=f"핫파트너스 - 무한 방지 목록 감지 알림",
            description=(
                f"temp_running_idx_list: {temp_running_idx_list}\n"
                f"에러 메세지: {err_msg}\n"
            ),
            format="error_consecutive_fail",
            tag_type='hotpartners-scheduler',
            slack_url=SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    async def send_slack_alert_unorderable_order_list(
        self,
        message,
        idx_list_str,
    ):
        button_list = [
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "일괄재시도",
                            "emoji": True
                        },
                        "value": "retry-button",  # 고유한 value
                        "url": f'https://api.hotpartners.co.kr/admin/orders/retry/{idx_list_str}',
                        "action_id": "retry-action"  # 고유한 action_id
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "일괄취소",
                            "emoji": True
                        },
                        "value": "cancel-button",  # 고유한 value
                        "url": "https://api.hotpartners.co.kr/admin/orders/bulk-cancel",
                        "action_id": "cancel-action"  # 고유한 action_id
                    }
                ]
            }
        ]

        await self.warning(
            # title=f"주문 불가 Pending 상태 주문 목록",
            title=f"댓글 생성 실패 주문 목록",
            description=f"{message}",
            button_list=button_list,
            tag_type='developers',
            format="error_consecutive_fail",
            slack_url=SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )


    async def send_slack_alert_is_pre_check_false_order_list(
        self,
        message,
        idx_list_str,
    ):
        await self.warning(
            title=f"주문 선검증 실패 Pending 상태 주문",
            description=(
                f"""{message}"""
            ),
            tag_type='hotpartners-scheduler',
            format="error_consecutive_fail",
            # button=[
            #     Button(title="일괄재시도", url=f'https://api.hotpartners.co.kr/admin/orders/retry/{idx_list_str}'),
            # ],
            slack_url=SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )


    async def send_slack_alert_order_cancel_monitor(
        self,
        message,
    ):
        await self.warning(
            title=f"파싱 연속 실패로 검증 체크 필요",
            description=(
                f"""{message}"""
            ),
            tag_type='decago',
            format="error_consecutive_fail",
            # button=[
            #     Button(title="일괄재시도", url=f'https://api.hotpartners.co.kr/admin/orders/retry/{idx_list_str}'),
            # ],
            slack_url=SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )


    async def send_slack_alert_developer_notification(
        self,
        message,
    ):
        await self.warning(
            title=f"다중 분배 주문 정보 (cf. 시범용)",
            description=(
                f"""{message}"""
            ),
            tag_type='hotpartners-scheduler',
            format="error_consecutive_fail",
            slack_url=SLACK_SERVICE_ERROR_WEBHOOK_URI,
        )

    async def send_panel_test_result(
        self,
        change_warning_logs,
        **kwargs,
    ) -> None:
        try:

            LOGGER.info(f"change_warning_logs: {change_warning_logs}")
            mentions = []
            if is_working_hours():
                mentions=[
                    'U07V7PA0DDZ',  # NOTE: 안수민 사원
                    'U04JBRN0M3K',  # NOTE: 송호진 주임
                ]
            task = asyncio.create_task(
                send_alarm_to_thread(
                    logger=LOGGER,
                    level=AlarmLevel.WARNING,
                    title=kwargs.get('title', ''),
                    channel_id=settings.SLACK_ORDER_SITE_BALANCE_CHANNEL_ID,  # 셀프마케팅플랫폼-잔고-알림방
                    messages=[f"{''.join(change_warning_logs)}"],
                    extra_blocks=[kwargs.get('button', [])],
                    mentions=mentions,
                    try_cnt=1,
                    split_message=True,
                )
            )

            await task

            if task.exception():
                raise task.exception()

        except Exception as e:
            LOGGER.error(
                f"""
                    슬랙 메세지 전송 실패:
                    error: {str(e)}
                """
            )