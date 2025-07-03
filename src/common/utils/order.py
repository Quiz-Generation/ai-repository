import re
from typing import Dict, List
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.common.utils.normalize import normalize_string

from ..constants.__init__ import Constants

# TODO:
"""
    19 는 내부용 -> 197xx && xx2xx 은 프로필 링크로 | 196xx 게시물 링크로
    ETC = 7
    ____ = 6 (인사이트)

    인스타그램 릴스 조회수 -> 19605

    + 링크를 위한 타입 컬럼 추가 및 서비스 수정시 설정 가능할게 (그룹단위로 ... ) 할것인지 ... 고민 필요
"""

# def _extract_instagram_username_from_url(url: str):
#     if url is None:
#         return "" # NOTE: 임시방편

#     # ex) https://www.instagram.com/95pje/
#     if "?" in url:  # ? 앞에 / 없는 경우때문에 먼저 처리
#         url = url[: url.index("?")]

#     # e.g. https://www.instagram.com/flowershop_mingo/profilecard/?igsh=aXVyZnRjbGd4NWI2
#     if "/profilecard/" in url:  # ? 앞에 / 없는 경우때문에 먼저 처리
#         url = url.split("/profilecard/")[0]

#     urls = url.split("/")
#     # URL 분할 후 유효한 세그먼트가 있는지 확인
#     if not urls or (len(urls) == 1 and not urls[0]):
#         return ""

#     for i in reversed(range(0, len(urls))):
#         if len(urls[i].strip()) > 0:
#             url = urls[i].strip()
#             break
#     if url and url[0] == '@':
#         url = url[1:]
#     # return url.lower()

#     return url
def _extract_instagram_username_from_url(url: str):
    if url is None:
        return "" # NOTE: 임시방편

    # URL에서 프로토콜 제거
    url = url.replace('https://', '').replace('http://', '')

    # 'instagram.com/' 제거
    if 'instagram.com/' in url:
        url = url.split('instagram.com/', 1)[-1]

    # '?' 이후의 쿼리 파라미터 제거
    if "?" in url:
        url = url.split("?", 1)[0]

    # '/profilecard/' 또는 '/profilecard' 이후 제거
    if "/profilecard/" in url:
        url = url.split("/profilecard/", 1)[0]
    if "/profilecard" in url:
        url = url.split("/profilecard", 1)[0]

    # '/' 로 분리된 경우 패턴에 맞게 처리
    pattern_list = ['p', 'reel', 'tv','reels', 'stories']
    url_split = url.strip('/').split('/')


    if any(p in url_split for p in pattern_list):
        url = url_split[-1]
    else:
        url = url_split[0]
    # e.g. 패턴케이스 : https://www.instagram.com/p/C48VOJ2p1e0/ -> ['p', 'C48VOJ2p1e0'] -> 'C48VOJ2p1e0'
    # e.g. 비패턴케이스 : https://www.instagram.com/yoojung_0101/. -> ['yoojung_0101', '.'] -> 'yoojung_0101'

    # '@' 문자로 시작하는 경우 제거
    if url.startswith('@'):
        url = url[1:]

    return url

def _extract_thread_id_from_url(url: str):
    if url is None:
        return ""

    # URL에서 프로토콜 제거
    url = url.replace('https://', '').replace('http://', '')

    # 'threads.net/' 제거
    if 'threads.net/' in url:
        url = url.split('threads.net/', 1)[-1]

    # '?' 이후의 쿼리 파라미터 제거
    if "?" in url:
        url = url.split("?", 1)[0]

    # '/' 로 분리된 경우 패턴에 맞게 처리
    url_split = url.strip('/').split('/')
    url = url_split[-1]
    # '@' 문자로 시작하는 경우 제거
    if url.startswith('@'):
        url = url[1:]
    return url


def _extract_instagram_shortcode_from_url(url: str):
    # URL에서 프로토콜 제거
    url = url.replace('https://', '').replace('http://', '')

    # 'instagram.com/' 제거
    if 'instagram.com/' in url:
        url = url.split('instagram.com/', 1)[-1]

    # '#advertiser' 제거
    if '#advertiser' in url:
        url = url.split('#advertiser', 1)[0]

    # e.g. https://www.instagram.com/p/C9Yr3Lvy0JH/c/17990637851677263/
    if "/c/" in url:
        return url

    # ex) https://www.instagram.com/p/COMeCkYlGo-/?utm_source=ig_web_copy_link
    if "?" in url:  # '?' 앞에 / 없는 경우때문에 먼저 처리
        url = url[: url.index("?")]

    if "#" in url:  # '#' 앞에 / 없는 경우때문에 먼저 처리
        url = url[: url.index("#")]
        if len(url) < 10:
            return ""

    # # '/profilecard/' 이후 제거
    # if "/profilecard/" in url:
    #     url = url.split("/profilecard/", 1)[0]
    # '/profilecard/' 또는 '/profilecard' 이후 제거
    if "/profilecard/" in url:
        url = url.split("/profilecard/", 1)[0]
    if "/profilecard" in url:
        url = url.split("/profilecard", 1)[0]

    if url and url[-1] == '/':
        url = url[:-1]

    urls = url.split("/")
    # '/' 로 분리된 경우 패턴에 맞게 처리
    pattern_list = ['p', 'reel', 'tv','reels', 'stories', 'share']
    url_split = url.strip('/').split('/')


    if any(p in url_split for p in pattern_list):
        url = url_split[-1]
    else:
        url = url_split[0]

    # URL 분할 후 유효한 세그먼트가 있는지 확인
    if not urls or (len(urls) == 1 and not urls[0]):
        return ""

    if urls[-1] == '':
        urls.pop()
    if 'profilecard' in urls:
        urls.remove('profilecard')

    for i in reversed(range(0, len(urls))):
        if len(urls[i].strip()) >= 11:
            url = urls[i].strip()
            break

    # NOTE: 2024-11-21 기준 shortcode 11자리로 제한 X
    # # e.g. https://www.instagram.com/p/C4VrYmmPwftc9w-2iMn4-3qYwm4YmYNS0H3NUc0/
    # if len(url) != 11:  # 11자리 아니면 오류. 단 가끔 긴 미디어 url 들어옴 -> short code로 바꾸기
    #     # 인덱스 -1과 -2 둘다 확인하는 이유는 앞단에서 링크 맨 뒤에 '/'를 무조건 짜르고 보내주는지 몰라서. 만약 있으면 urls[-1]는 ''이고 그게 아니면 코드 부분일 것.
    #     if len(urls[-1]) > 11:
    #         short_code = urls[-1][:11]
    #         return short_code
    #     return ""
    # else:
    #     return url

    return url


async def _extract_youtube_link(link):
    url_elem = urlparse(link)
    url_query_map = dict(parse_qsl(url_elem.query))

    new_query = urlencode({k: v for k, v in url_query_map.items()})

    new_path = url_elem.path
    new_path = re.sub(r'^/', '', new_path)

    new_url = urlunparse(
        url_elem._replace(
            scheme="",  # https:// 제거
            netloc="",  # *.youtube.com/ 제거
            path=new_path,  # 파라미터로 새로운 Path 값이 넘어오는 경우 해당 값으로 교체
            params=url_elem.params,
            query=new_query,  # 새로운 쿼리 적용
        )
    )

    return new_url

def extract_hashtag_and_link(link):
    """
    링크에서 해시태그와 링크를 추출하는 함수

    e.g. 10801 주문 링크 (#위스키|https://www.instagram.com/p/CzU06shBHpE/)
    """
    search_tag = ''

    if '|' in link:
        try:
            search_tag, link = link.replace(' ', '').split('|', 1)

            # 추출된 해시태그가 있을 경우
            if search_tag:
                hashtag_list = re.findall(r'#\w+', search_tag)

                # 첫 번째 해시태그만 사용
                for hashtag in hashtag_list:
                    search_tag = hashtag
                    break
        except ValueError:
            pass

    return search_tag, link

def _extract_facebook_post_url_from_url(
    link: str,
    general_type
) -> str:
    """페이스북 게시글 URL 에서 파싱시 필요한 고유값을 추출 합니다.
        - 스토리 게시글 (내용 + 여러 사진)
            - www.facebook.com/<계정고유값>/posts/<게시글고유ID>
                - 위 URL path에 기재된 값 외는 없어도 됨
            - m.facebook.com/story.php?story_fbid=<게시글고유ID>&id=<계정고유값>
                - 'story_fbid', 'id' 쿼리 필수 (이외 쿼리는 없어도 됨)
            - m.facebook.com/permalink.php?story_fbid=<게시글고유ID>&id=<계정고유값>
                - 'story_fbid', 'id' 쿼리 필수 (이외 쿼리는 없어도 됨)

            └--> 타임피드에 보이는 게시글 (날짜) 클릭시 위 포맷으로 접근
                두개 형식 모두 모바일/피씨에서 혼용 가능

        - 사진 게시글 (스토리 게시글의 사진클릭)
            - www.facebook.com/photo?fbid=<사진게시글고유ID>
            - m.facebook.com/photo.php?fbid=<사진게시글고유ID>&id=<계정고유값>
                - 'fbid' 쿼리 필수 (이외 쿼리는 없어도 됨)

            - www(m).facebook.com/<계정고유값>/photos/<게시글고유ID>/<선택사진ID>
                - 위 URL path에 기재된 값 외는 없어도 됨

            - m.facebook.com/<계정고유값>/posts/<게시글고유ID>?photo_id=<선택사진ID>
                - 해당 포맷은 가끔식 나옴..
                - 'photo_id' 쿼리가 없으면 게시글 처음부터 보이고, 있으면 해당 사진 부터 보임
                    - ex) ('https://m.facebook.com/eminem/posts/'
                           'pcb.608991013928713?photo_id=608985010595980')

            └--> 스토리 게시글 내에 존재하는 '특정' 사진 클릭시 위 포맷으로 접근
                 두개 형식 모두 모바일/피씨에서 혼용 가능

        - 영상 게시글
            - www.facebook.com/<사용자계정고유값>/videos/<영상게시글고유ID>
            - www(m).facebook.com/watch/?v=387615970104956
            - www.facebook.com/reel/3393625447533007
            └--> 위 방식인 경우, 비디오고유ID 를 사용하여, 모두 story 형태로 처리 가능

    :return: str
        XXXXX
            -> 'photo?fbid=<사진게시글고유ID>'

    """

    class URLType(object):
        STORY = "story"  # or 'story.php'
        PERMALINK = "permalink"  # or 'permalink.php
        POSTS = "posts"
        PHOTO = "photo"  # or 'photo.php'
        PHOTOS = "photos"
        VIDEOS = "videos"  # 지원하지 않음
        WATCH = "watch"  # 지원하지 않음
        REEL = "reel"
        FB_WATCH = "fb.watch"

    def _filter_url(
        new_path: str = "",  # 대체할 새로운 URL Path
        new_query: Dict = {},  # 추가할 쿼리 정보
        required_query: List[str] = [],  # 기존 쿼리에서 필요한 쿼리 Key 값
    ) -> str:
        query = {
            k: v for k, v in url_query_map.items() if k in required_query
        }
        if new_query:
            query.update(new_query)

        path = url_elem.path
        path = re.sub(r'^/', '', path)

        if new_path:
            path = re.sub(r'^/', '', new_path)


        return urlunparse(
            url_elem._replace(
                scheme="",  # https:// 제거
                netloc="",  # *.facebook.com/ 제거
                path=path,  # 파라미터로 새로운 Path 값이 넘어오는 경우 해당 값으로 교체
                query=urlencode(query)  # 새로운 쿼리 적용
            )
        )
    if 'groups/' in link:
        link = link.replace('groups/','')

    # TODO: 데카고에 확인 요청 드림
    # if 'share/p/' in link:
    #     # e.g. 'https://www.facebook.com/share/p/rfosTEVDQRhjEPmc/' -> 'https://www.facebook.com/rfosTEVDQRhjEPmc/'
    #     link = link.replace('share/p/','')
    #     if link[-1] == "/":
    #         link = link[:-1]

    url_elem = urlparse(link)
    url_path = url_elem.path
    url_path_list = [u for u in url_path.split("/") if u]
    url_query_map = dict(parse_qsl(url_elem.query))
    new_url = ''

    # -- 동영상 게시글 --
    if URLType.WATCH in url_path:
        # 'watch' 형태로 넘어오는 경우 그대로 반환
        #   - /watch?v=<비디오ID>

        if general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            new_url = _filter_url(
                new_path=url_path,
                required_query=['comment_id'],
            )

            if "comment_id" not in urlparse(new_url).query:
                return ""

            return new_url

        video_id = url_query_map.get("v")

        if video_id:
            new_url = _filter_url(
                new_path=f"reel/{video_id}",
                required_query=[],
            )

            return new_url

        return ""

    if URLType.VIDEOS in url_path or URLType.REEL in url_path:
        # 'video' 형태로 넘어오는 경우 비디오 ID 를 추출하여 watch 형태로 반환
        #   - /eminem/videos/<비디오ID>/
        # 'reel' 형태로 넘어오는 경우 비디오 ID를 추출하여 watch 형태로 반환
        #   - /reel/<비디오ID>/

        if general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            new_url = _filter_url(
                new_path=url_path,
                required_query=['comment_id'],
            )

            if "comment_id" not in urlparse(new_url).query:
                return ""

            return new_url

        video_id = url_path_list[-1]

        if video_id:
            new_url = _filter_url(
                new_path=f"reel/{video_id}",
                required_query=[],
            )

            return new_url

        return ""

    # -- 스토리 게시글 --
    if (
        URLType.STORY in url_path
        and "story_fbid" in url_query_map
        and "id" in url_query_map
    ):
        # 'story' 형태로 넘어오는 경우 posts로 반환 - 프론트 딥링크 이슈로 posts로 변경
        #   - /story.php?story_fbid=<게시글고유ID>&id=<계정고유값>
        #   -> <계정고유값>/posts/<게시글고유ID>
        new_url = _filter_url(
            new_path=f"{url_query_map['id']}/posts/{url_query_map['story_fbid']}",
            required_query=[
                "m_entstream_source",  # 영상 게시글인 경우, watch 형태로 redirect 안되게
                "comment_id",  # 댓글 고유 ID
            ],
        )
        if general_type == Constants.Service.GeneralType.Facebook.COMMENT:
            if "comment_id" in urlparse(new_url).query:
                return ""

        elif general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            if "comment_id" not in urlparse(new_url).query:
                return ""

        return new_url

    if URLType.POSTS in url_path and "photo_id" not in url_query_map:
        new_url = _filter_url(
            new_path="",
            required_query=[
                "comment_id",  # 댓글 고유 ID
            ],
        )
        if general_type == Constants.Service.GeneralType.Facebook.COMMENT:
            if "comment_id" in urlparse(new_url).query:
                return ""

        elif general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            if "comment_id" not in urlparse(new_url).query:
                return ""

        return new_url

    if (
        URLType.PERMALINK in url_path
        and "story_fbid" in url_query_map
        and "id" in url_query_map
    ):
        # 'permalink' 형태로 넘어오는 경우, 'posts' 형태로 변환 후 반환 - 프론트 딥링크 이슈로 posts로 변경
        #   - /permalink.php?story_fbid=<게시글고유ID>&id=<계정고유값>
        #   -> <게시글고유ID>/posts/<게시글고유ID>
        new_url = _filter_url(
            new_path=f"{url_query_map['id']}/posts/{url_query_map['story_fbid']}",
            required_query=[
                "m_entstream_source",  # 영상 게시글인 경우, watch 형태로 redirect 안되게
                "comment_id",  # 댓글 고유 ID
            ],
        )
        if general_type == Constants.Service.GeneralType.Facebook.COMMENT:
            if "comment_id" in urlparse(new_url).query:
                return ""

        elif general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            if "comment_id" not in urlparse(new_url).query:
                return ""

        return new_url

    # -- 사진 게시글 --
    if URLType.PHOTOS in url_path:
        # 'photos' 형태로 넘어오는 경우 불필요한 쿼리는 모두 제거하고 사용
        #  - /<계정고유값>/photos/<게시글고유ID>/<선택사진ID>
        new_url = _filter_url(
            required_query=[
                "comment_id",  # 댓글 고유 ID
            ],
        )
        if general_type == Constants.Service.GeneralType.Facebook.COMMENT:
            if "comment_id" in urlparse(new_url).query:
                return ""

        elif general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            if "comment_id" not in urlparse(new_url).query:
                return ""


        return new_url

    if URLType.PHOTO in url_path:
        # 'photo' 형태로 넘어오는 경우 필수 쿼리를 제외하고 반환
        #  - /photo.php?fbid=<사진게시글고유ID>&id=<계정고유값>
        new_url = _filter_url(
            new_path="photo.php",
            required_query=[
                "fbid",
                "id",
                "comment_id",  # 댓글 고유 ID
            ],
        )

        if general_type == Constants.Service.GeneralType.Facebook.COMMENT:
            if "comment_id" in urlparse(new_url).query:
                return ""

        elif general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            if "comment_id" not in urlparse(new_url).query:
                return ""


        return new_url

    if URLType.POSTS in url_path and "photo_id" in url_query_map:
        # 'posts' 형태에 'photo_id' 쿼리가 추가되어 오는 경우, 'photos' 형태로 변경 후 반환
        #  - /<계정고유값>/posts/<게시글고유ID>?photo_id=<선택사진ID>
        #    => /<계정고유값>/photos/<게시글고유ID>/<선택사진ID>
        user_unique_id_or_account = url_path_list[0]
        media_unique_id = url_path_list[2]
        media_photo_unique_id = url_query_map["photo_id"]
        new_url = _filter_url(
            new_path=(
                f"{user_unique_id_or_account}/photos/"
                f"{media_unique_id}/{media_photo_unique_id}"
            ),
            required_query=[
                "comment_id",  # 댓글 고유 ID
            ],
        )

        if general_type == Constants.Service.GeneralType.Facebook.COMMENT:
            if "comment_id" in urlparse(new_url).query:
                return ""

        elif general_type == Constants.Service.GeneralType.Facebook.COMMENT_LIKE:
            if "comment_id" not in urlparse(new_url).query:
                return ""

        return new_url

    if URLType.FB_WATCH in url_elem.netloc:
        new_url = _filter_url(
            new_path=(f"{url_elem.netloc}{url_elem.path}")
        )
        return new_url

    new_url = _filter_url()
    return new_url

def _extract_facebook_unique_id_from_url(link: str) -> str | None:
    """페이스북 유형에 따른 사용자 프로필 / 페이지 URL 에서 고유값을 추출 합니다.
        - 프로필 또는 페이지
            - URL 형태 1
                - m.facebook.com/profile.php?id=<고유 값>
                - m.facebook.com/profile.php?id=<고유 값>&ref=XXXX

                - m.facebook.com/profile.php?id=<고유 값>&v=info
                - m.facebook.com/profile.php?id=<고유 값>&v=info&lst=XXXX

            - URL 형태 2
                - m.facebook.com/<고유 계정 이름 (영어)>
                - m.facebook.com/<고유 계정 이름 (영어)>?ref=XXXX

                - m.facebook.com/<고유 계정 이름 (영어)>/about
                - m.facebook.com/<고유 계정 이름 (영어)>/about?lst=XXXX
                - m.facebook.com/<고유 계정 이름 (영어)>?v=info

        - 그룹 (미지원)
            - URL 형태: m.facebook.com/groups/<고유 값>/

    :return: str
        XXXXX
            -> 0
        XXXXX
            -> 1

    """
    # NOTE: 왜 비워둔 함수인지 모르겠지만, 혹시 몰라 아래와 같은 코드를 추가하고 주석 처리함
    # parsed_url = urlparse(link)
    # path = parsed_url.path.strip('/')
    # query = parse_qs(parsed_url.query)

    # # URL 형태 1: profile.php?id=<고유 값>
    # if 'profile.php' in path:
    #     return query.get('id', [None])[0]

    # # URL 형태 2: /<고유 계정 이름 (영어)>
    # elif path and '/' not in path:
    #     return path

    # # URL 형태 2의 변형: /<고유 계정 이름 (영어)>/about 등
    # elif path and '/' in path:
    #     return path.split('/')[0]

    # # 그룹 URL (미지원이지만 처리)
    # elif path.startswith('groups/'):
    #     return None

    # # 기타 경우
    # else:
    #     return None

    # # # 테스트
    # # test_urls = [
    # #     "https://m.facebook.com/profile.php?id=100085131144633",
    # #     "https://m.facebook.com/profile.php?id=100085131144633&ref=XXXX",
    # #     "https://m.facebook.com/profile.php?id=100085131144633&v=info",
    # #     "https://m.facebook.com/zuck",
    # #     "https://m.facebook.com/zuck?ref=XXXX",
    # #     "https://m.facebook.com/zuck/about",
    # #     "https://m.facebook.com/zuck/about?lst=XXXX",
    # #     "https://m.facebook.com/zuck?v=info",
    # #     "https://m.facebook.com/groups/123456789/",
    # #     "https://www.facebook.com/share/1Ak8AbBrsU/?mibextid=wwXIfr"
    # # ]

    # # for url in test_urls:
    # #     print(f"URL: {url}")
    # #     print(f"Extracted ID: {_extract_facebook_unique_id_from_url(url)}\n")

    # # """
    # # 각 URL에 대한 추출 결과는 아래놔 같다:
    # #     "100085131144633"
    # #     "100085131144633"
    # #     "100085131144633"
    # #     "zuck"
    # #     "zuck"
    # #     "zuck"
    # #     "zuck"
    # #     "zuck"
    # #     None (그룹 URL은 미지원)
    # #     profile.php?id= 형식의 URL에서는 ID 번호를 추출
    # #     사용자명 형식의 URL에서는 사용자명을 추출.
    # #     추가적인 경로나 쿼리 파라미터가 있어도 올바르게 고유값을 추출
    # #     그룹 URL의 경우 None을 반환하여 미지원임을 나타냄
    # # """

    class PathType(object):
        PROFILE = "profile.php"
        SHARE = "share"

    url_elem = urlparse(link)
    url_query_map = dict(parse_qsl(url_elem.query))

    fb_unique_str: str | None = None

    if PathType.PROFILE in url_elem.path:
        # /profile.php?id=XXXX 형태로 들어온 경우, id 쿼리값만 추출
        url_query = url_elem.query
        url_query_elem: Dict = dict(parse_qsl(url_query))
        fb_unique_str = url_query_elem.get("id")

    elif PathType.SHARE in url_elem.path:
        # /share/ 형태로 들어온 경우
        return link
        # fb_unique_str = url_elem.path + url_elem.params + url_elem.query
        # fb_unique_str = url_elem.path.replace('/share/', '') + url_elem.params + url_elem.query
        # fb_unique_str = url_elem.path.replace('/share/', '') # '1Ak8AbBrsU/'
        # if fb_unique_str and fb_unique_str[-1] == '/':
        #     fb_unique_str = fb_unique_str[:-1]
    else:
        url_path_list: List[str] = [u for u in (url_elem.path).split("/") if u]
        if url_path_list:
            fb_unique_str = url_path_list[0]

    return fb_unique_str


def extract_processed_link(
    original_link: str,
    category: str,
    general_type: str,
    service: str,
) -> str | None:
    normalized_original_link: str = normalize_string(original_link)

    link: str | None = normalized_original_link

    if category == Constants.Category.INSTAGRAM:
        if (
            general_type == Constants.Service.GeneralType.Instagram.FOLLOW
            or (
                general_type == Constants.Service.GeneralType.Instagram.ETC
                and str(service)[-4] == '9'
            )
            or
            general_type == Constants.Service.GeneralType.Instagram.ACCOUNT
        ):
            link = _extract_instagram_username_from_url(link)
        else:
            link = _extract_instagram_shortcode_from_url(link)
    elif category == Constants.Category.FACEBOOK:
        if (
            general_type == Constants.Service.GeneralType.Facebook.PAGELIKE
            or general_type == Constants.Service.GeneralType.Facebook.FOLLOW
        ):
            link = _extract_facebook_unique_id_from_url(link)
        else:
            link = _extract_facebook_post_url_from_url(link, general_type)
    elif category == Constants.Category.THREAD:
        if general_type == Constants.Service.GeneralType.Thread.FOLLOW:
            link = _extract_thread_id_from_url(link)
        else:
            # TODO: 지금은 하단 로직 타는 모든 주문들은 파싱 안태우게 외부 패널로 보내고 있음. 나중에 내부적으로 처리할 때는 하단 함수는 아닐듯 ... 히스토리 체크 필요
            link = _extract_instagram_shortcode_from_url(link)
    elif category == Constants.Category.YOUTUBE:
        # NOTE.
        # 241206 현재 유튜브 주문은 외부 패널로 보내고 있음. 나중에 내부적으로 처리할 때는 하단 함수는 아닐듯 ... 히스토리 체크 필요 (w. 김정환)
        # link = _extract_youtube_link(link)
        pass
    else:
        pass

    if not link:
        return normalized_original_link

    return link