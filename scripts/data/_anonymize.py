import re

RE_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
RE_ID = re.compile(r"(?<!\d)\d{15}(?:\d{2}[0-9Xx])?(?!\d)")
RE_STU_ID = re.compile(r"(?<!\d)\d{8,12}(?!\d)")
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_QQ = re.compile(r"(?i)(?:qq|微信|wechat|wx)[：:\s]*\d{5,12}")
RE_NAME = re.compile(r"(我叫|我是|本人叫)[\u4e00-\u9fa5]{2,4}(?=[，。,.\s])")


def anonymize(text: str) -> str:
    # phone/id before generic long-digit, else stu-id would eat those
    t = RE_PHONE.sub("<PHONE>", text)
    t = RE_ID.sub("<ID>", t)
    t = RE_QQ.sub("<SOCIAL>", t)
    t = RE_STU_ID.sub("<NUM>", t)
    t = RE_EMAIL.sub("<EMAIL>", t)
    t = RE_NAME.sub("我是某同学", t)
    return t
