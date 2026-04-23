"""Email reading tool via IMAP."""

import imaplib
import email
import os
from email.header import decode_header
from datetime import datetime

from langchain.tools import tool

def _get_required_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        raise ValueError(f"缺少环境变量 {name}，请在 .env 中配置。")
    return value


def _decode_str(raw: str) -> str:
    """解码邮件头中的编码字符串（如 =?utf-8?B?...?=）。"""
    if not raw:
        return ""
    parts = decode_header(raw)
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded)


def _parse_email(msg: email.message.Message) -> dict:
    """解析单封邮件为结构化字典。"""
    subject = _decode_str(msg.get("Subject", ""))
    from_addr = _decode_str(msg.get("From", ""))
    date_str = msg.get("Date", "")
    try:
        dt = email.utils.parsedate_to_datetime(date_str)
        date_formatted = dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        date_formatted = date_str

    # 获取邮件正文（纯文本优先）
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                body = payload.decode(charset, errors="replace")
                break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body = payload.decode(charset, errors="replace")

    # 限制正文长度
    body_preview = body[:500].replace("\n", " ").strip()

    return {
        "subject": subject,
        "from": from_addr,
        "date": date_formatted,
        "body": body_preview,
    }


@tool
def read_emails(max_count: int = 5, keyword: str = "") -> str:
    """读取最近收到的邮件。

    参数:
        max_count: 最多返回多少封邮件，默认 5。
        keyword: 可选，只返回主题或发件人包含该关键词的邮件。

    返回每封邮件的发件人、主题、时间和正文摘要。
    """
    try:
        imap_server = os.getenv("IMAP_SERVER", "imap.qq.com")
        imap_port = int(os.getenv("IMAP_PORT", "993"))
        imap_user = _get_required_env("IMAP_USER")
        imap_password = _get_required_env("IMAP_PASSWORD")
    except Exception as e:
        return f"邮件工具未配置：{e}"

    try:
        with imaplib.IMAP4_SSL(imap_server, imap_port) as mail:
            mail.login(imap_user, imap_password)
            mail.select("INBOX")

            # 搜索邮件
            if keyword:
                status, data = mail.search(None, "ALL")
            else:
                status, data = mail.search(None, "ALL")

            if status != "OK":
                return "邮件服务器返回异常。"

            msg_ids = data[0].split()
            if not msg_ids:
                return "收件箱中没有邮件。"

            # 取最近 max_count 封（倒序）
            recent_ids = msg_ids[-max_count:][::-1]

            results = []
            for mid in recent_ids:
                status, msg_data = mail.fetch(mid, "(RFC822)")
                if status != "OK":
                    continue
                raw_msg = email.message_from_bytes(msg_data[0][1])
                parsed = _parse_email(raw_msg)

                # 关键词过滤
                if keyword:
                    if keyword.lower() not in parsed["subject"].lower() and \
                       keyword.lower() not in parsed["from"].lower():
                        continue

                results.append(
                    f"发件人: {parsed['from']}\n"
                    f"主题: {parsed['subject']}\n"
                    f"时间: {parsed['date']}\n"
                    f"正文: {parsed['body']}"
                )

            if not results:
                return f"没有找到匹配「{keyword}」的邮件。"

            return "\n\n---\n\n".join(results)

    except imaplib.IMAP4.error as e:
        return f"邮件登录失败: {e}"
    except Exception as e:
        return f"读取邮件出错: {e}"
