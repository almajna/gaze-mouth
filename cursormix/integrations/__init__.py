from .base import Integration
from .mailto import MailDraft

SERVICES: dict[str, Integration] = {"compose_mail": MailDraft()}
