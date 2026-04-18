"""Email pytest results to the configured recipient via SMTP.

Reads SMTP/recipient config from environment variables (the same ones used by
.github/workflows/daily-report.yml and config.py). Intended to run as a step
in the `tests.yml` workflow.

Sends a mobile-friendly HTML body (with a plain-text fallback) and attaches
the JUnit XML report. Skips silently if SMTP config is missing so forks
without secrets don't fail.
"""

from __future__ import annotations

import html
import os
import re
import smtplib
import sys
import xml.etree.ElementTree as ET
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


OUTPUT_FILE = Path("pytest-output.txt")
JUNIT_FILE = Path("pytest-report.xml")
MAX_BODY_LINES = 200


def _required(name: str) -> str | None:
    val = os.environ.get(name, "").strip()
    return val or None


def _parse_junit_summary() -> dict:
    """Return aggregated counts and per-test status from pytest-report.xml."""
    summary = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0, "time": 0.0, "cases": []}
    if not JUNIT_FILE.exists():
        return summary
    try:
        root = ET.parse(JUNIT_FILE).getroot()
    except ET.ParseError:
        return summary

    suites = root.findall(".//testsuite") or [root]
    for suite in suites:
        summary["total"] += int(suite.get("tests", 0))
        summary["failed"] += int(suite.get("failures", 0))
        summary["errors"] += int(suite.get("errors", 0))
        summary["skipped"] += int(suite.get("skipped", 0))
        summary["time"] += float(suite.get("time", 0.0) or 0.0)
        for case in suite.findall("testcase"):
            status = "passed"
            if case.find("failure") is not None:
                status = "failed"
            elif case.find("error") is not None:
                status = "error"
            elif case.find("skipped") is not None:
                status = "skipped"
            summary["cases"].append({
                "classname": case.get("classname", ""),
                "name": case.get("name", ""),
                "status": status,
                "time": float(case.get("time", 0.0) or 0.0),
            })
    summary["passed"] = summary["total"] - summary["failed"] - summary["errors"] - summary["skipped"]
    return summary


def _tail_output() -> tuple[str, int, int]:
    if not OUTPUT_FILE.exists():
        return "(pytest output file not found)", 0, 0
    lines = OUTPUT_FILE.read_text(errors="replace").splitlines()
    tail = lines[-MAX_BODY_LINES:]
    return "\n".join(tail), len(tail), len(lines)


def _build_text_body(outcome: str, summary: dict) -> str:
    repo = os.environ.get("GH_REPO", "unknown")
    ref = os.environ.get("GH_REF", "unknown")
    sha = os.environ.get("GH_SHA", "unknown")[:7]
    run_url = os.environ.get("GH_RUN_URL", "")
    excerpt, shown, total_lines = _tail_output()
    prefix = f"(showing last {shown} of {total_lines} lines)\n\n" if total_lines > shown else ""
    return (
        f"Repository: {repo}\n"
        f"Branch:     {ref}\n"
        f"Commit:     {sha}\n"
        f"Outcome:    {outcome.upper()}\n"
        f"Tests:      {summary['passed']}/{summary['total']} passed "
        f"(failed={summary['failed']}, errors={summary['errors']}, skipped={summary['skipped']})\n"
        f"Duration:   {summary['time']:.2f}s\n"
        f"Run:        {run_url}\n"
        f"\n---- pytest output ----\n{prefix}{excerpt}\n"
    )


_STATUS_COLORS = {
    "passed": "#2f8f2a",
    "failed": "#c1272d",
    "error": "#c1272d",
    "skipped": "#b58900",
}


def _ansi_strip(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _build_html_body(outcome: str, summary: dict) -> str:
    repo = html.escape(os.environ.get("GH_REPO", "unknown"))
    ref = html.escape(os.environ.get("GH_REF", "unknown"))
    sha = html.escape(os.environ.get("GH_SHA", "unknown")[:7])
    run_url = html.escape(os.environ.get("GH_RUN_URL", ""), quote=True)

    is_pass = outcome == "success" and summary["failed"] == 0 and summary["errors"] == 0
    banner_color = "#2f8f2a" if is_pass else "#c1272d"
    banner_text = "PASS" if is_pass else "FAIL"

    excerpt, shown, total_lines = _tail_output()
    excerpt = _ansi_strip(excerpt)
    truncation = (
        f'<p style="color:#666;margin:0 0 8px 0;font-size:13px;">'
        f"Showing last {shown} of {total_lines} lines."
        f"</p>"
        if total_lines > shown
        else ""
    )

    failed_rows = ""
    failures = [c for c in summary["cases"] if c["status"] in ("failed", "error")]
    if failures:
        rows = "".join(
            f'<tr><td style="padding:6px 10px;border-bottom:1px solid #eee;">'
            f'<code style="font-family:ui-monospace,Menlo,Consolas,monospace;color:#c1272d;">'
            f'{html.escape(c["classname"])}::{html.escape(c["name"])}'
            f"</code></td><td style=\"padding:6px 10px;text-align:right;color:#c1272d;\">"
            f'{c["status"].upper()}</td></tr>'
            for c in failures
        )
        failed_rows = (
            '<h3 style="margin:24px 0 8px 0;font-size:16px;">Failures</h3>'
            '<table role="presentation" style="width:100%;border-collapse:collapse;'
            'background:#fff6f6;border:1px solid #f0caca;border-radius:6px;">'
            f"{rows}</table>"
        )

    return f"""<!DOCTYPE html>
<html>
  <body style="margin:0;padding:16px;background:#f5f5f0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#222;">
    <div style="max-width:720px;margin:0 auto;background:#ffffff;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.06);">
      <div style="background:{banner_color};color:#ffffff;padding:16px 20px;">
        <div style="font-size:13px;opacity:0.85;letter-spacing:0.5px;">PYTEST RESULT</div>
        <div style="font-size:28px;font-weight:700;margin-top:4px;">{banner_text}</div>
        <div style="font-size:15px;margin-top:4px;">
          {summary['passed']}/{summary['total']} passed · {summary['time']:.2f}s
        </div>
      </div>
      <div style="padding:20px;">
        <table role="presentation" style="width:100%;border-collapse:collapse;font-size:14px;">
          <tr><td style="padding:4px 0;color:#666;width:110px;">Repository</td>
              <td style="padding:4px 0;"><code style="font-family:ui-monospace,Menlo,Consolas,monospace;">{repo}</code></td></tr>
          <tr><td style="padding:4px 0;color:#666;">Branch</td>
              <td style="padding:4px 0;"><code style="font-family:ui-monospace,Menlo,Consolas,monospace;">{ref}</code></td></tr>
          <tr><td style="padding:4px 0;color:#666;">Commit</td>
              <td style="padding:4px 0;"><code style="font-family:ui-monospace,Menlo,Consolas,monospace;">{sha}</code></td></tr>
          <tr><td style="padding:4px 0;color:#666;">Failed</td>
              <td style="padding:4px 0;color:{'#c1272d' if summary['failed'] or summary['errors'] else '#222'};">
                  {summary['failed']} failed, {summary['errors']} errors, {summary['skipped']} skipped
              </td></tr>
          <tr><td style="padding:4px 0;color:#666;">Run</td>
              <td style="padding:4px 0;">
                  <a href="{run_url}" style="color:#003153;text-decoration:none;font-weight:600;">
                      View in GitHub Actions &rarr;
                  </a>
              </td></tr>
        </table>

        {failed_rows}

        <h3 style="margin:24px 0 8px 0;font-size:16px;">Pytest output</h3>
        {truncation}
        <pre style="background:#0f172a;color:#e2e8f0;padding:14px;border-radius:6px;
                    font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;
                    line-height:1.45;white-space:pre-wrap;word-break:break-word;
                    overflow-x:auto;margin:0;">{html.escape(excerpt)}</pre>
      </div>
      <div style="padding:12px 20px;background:#fafafa;color:#999;font-size:12px;text-align:center;">
        Sent from <code>.github/workflows/tests.yml</code> · JUnit XML attached.
      </div>
    </div>
  </body>
</html>
"""


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
    summary = _parse_junit_summary()
    is_pass = outcome == "success" and summary["failed"] == 0 and summary["errors"] == 0
    status_tag = "PASS" if is_pass else "FAIL"
    repo = os.environ.get("GH_REPO", "tests")
    ref = os.environ.get("GH_REF", "")

    msg = MIMEMultipart("alternative")
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = f"[{status_tag}] pytest — {repo}@{ref}"

    msg.attach(MIMEText(_build_text_body(outcome, summary), "plain", "utf-8"))
    msg.attach(MIMEText(_build_html_body(outcome, summary), "html", "utf-8"))

    if JUNIT_FILE.exists():
        with JUNIT_FILE.open("rb") as fh:
            part = MIMEApplication(fh.read(), _subtype="xml")
        part.add_header("Content-Disposition", "attachment", filename=JUNIT_FILE.name)
        # Attachments on a 'multipart/alternative' aren't well-formed; wrap
        # the alternative in a 'mixed' container.
        outer = MIMEMultipart("mixed")
        for k, v in msg.items():
            outer[k] = v
        outer.attach(msg)
        outer.attach(part)
        msg = outer

    try:
        with smtplib.SMTP(smtp_host, int(smtp_port), timeout=30) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
    except Exception as e:
        print(f"[email_test_results] SMTP send failed: {e}", file=sys.stderr)
        return 0

    print(f"[email_test_results] Sent test results to {email_to} ({status_tag}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
