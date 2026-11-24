from email.headerregistry import Address
from urllib.parse import quote, urlencode


class MailDraft:
    def prepare(self, arguments):
        if not isinstance(arguments, dict) or set(arguments) - {"to", "subject", "body"}:
            raise ValueError("Expected to, subject and body for the email draft.")
        to, subject, body = (arguments.get(k, "") for k in ("to", "subject", "body"))
        if not all(isinstance(x, str) for x in (to, subject, body)) or any(c in to + subject for c in "\r\n"):
            raise ValueError("Invalid email address or subject.")
        if to and not Address(addr_spec=to).domain:
            raise ValueError("Enter the recipient’s full email address.")
        return (
            "mailto:"
            + quote(to, safe="@")
            + "?"
            + urlencode({"subject": subject, "body": body}, quote_via=quote)
        )
