"""Email pytest results to the configured recipient via SMTP.

Reads SMTP/recipient config from environment variables (the same ones used by
.github/workflows/daily-report.yml and config.py). Intended to run as a step
in the `tests.yml` workflow.

Skips silently if SMTP config is missing so forks without secrets don't fail.
"""

from __future__ import annotations

import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path


OUTPUT_FILE = Path("pytest-output.txt")
JUNIT_FILE = Path("pytest-report.xml")
MAX_BODY_LINES = 200


def _required(name: str) -> str | None:
    val = os.environ.get(name, "").strip()
    return val or None


def _build_body(outcome: str) -> str:
    repo = os.environ.get("GH_REPO", "unknown")
    ref = os.environ.get("GH_REF", "unknown")
    sha = os.environ.get("GH_SHA", "unknown")[:7]
    run_url = os.environ.get("GH_RUN_URL", "")

    header = (
        f"Repository: {repo}\n"
        f"Branch:     {ref}\n"
        f"Commit:     {sha}\n"
        f"Outcome:    {outcome.upper()}\n"
        f"Run:        {run_url}\n"
    )

    if OUTPUT_FILE.exists():
        lines = OUTPUT_FILE.read_text(errors="replace").splitlines()
        tail = lines[-MAX_BODY_LINES:]
        truncated = len(lines) > MAX_BODY_LINES
        excerpt = "\n".join(tail)
        prefix = f"(showing last {MAX_BODY_LINES} of {len(lines)} lines)\n\n" if truncated else ""
        body_tail = f"\n\n---- pytest output ----\n{prefix}{excerpt}\n"
    else:
        body_tail = "\n\n(pytest output file not found)\n"

    return header + body_tail


def main() -> int:
    email_to = _required("EMAIL_TO")
    email_from = _required("EMAIL_FROM")
    smtp_host = _required("SMTP_HOST")
    smtp_port = _required("SMTP_PORT")
    smtp_user = _required("SMTP_USER")
    smtp_password = _required("SMTP_PASSWORD")

    if not all([email_to, email_from, smtp_host, smtp_port, smtp_user, smtp_password]):
        print("[email_test_results] SMTP config incomplete, skipping email.")
        return 0

    outcome = os.environ.get("PYTEST_OUTCOME", "unknown")
    status_tag = "PASS" if outcome == "success" else "FAIL"
    repo = os.environ.get("GH_REPO", "tests")
    ref = os.environ.get("GH_REF", "")

    msg = MIMEMultipart()
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = f"[{status_tag}] pytest — {repo}@{ref}"

    msg.attach(MIMEText(_build_body(outcome), "plain"))

    if JUNIT_FILE.exists():
        with JUNIT_FILE.open("rb") as fh:
            part = MIMEApplication(fh.read(), _subtype="xml")
        part.add_header(
            "Content-Disposition", "attachment", filename=JUNIT_FILE.name
        )
        msg.attach(part)

    try:
        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    except Exception as e:
        # Don't fail the job solely because email delivery failed — the
        # pytest step already controls the overall success/failure.
        print(f"[email_test_results] SMTP send failed: {e}", file=sys.stderr)
        return 0

    print(f"[email_test_results] Sent test results to {email_to} ({status_tag}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
