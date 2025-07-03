import unicodedata


def normalize_string(text):
    NORMALIZATION_FORM = 'NFKC'
    try:
        return unicodedata.normalize(NORMALIZATION_FORM, text)
    except TypeError:
        raise TypeError(f"입력은 반드시 문자열이어야 합니다.\n입력값: {text}")
    except Exception as e:
        raise (f"정규화 중 예상치 못한 오류 발생: {str(e)}")
