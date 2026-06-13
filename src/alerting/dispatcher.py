"""Multi-channel alert dispatcher — webhook, Slack, PagerDuty, email.

Each channel implements the AlertChannel protocol. Dispatching is
async with per-channel circuit breaker protection.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import httpx

from src.alerting.models import Alert
from src.core.circuit_breaker import get_breaker
from src.core.config import get_config
from src.core.logging import get_logger

logger = get_logger("alerting.dispatcher")


class AlertChannel(ABC):
    """Abstract alert channel interface."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """Send alert to this channel. Returns True on success."""
        ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class WebhookChannel(AlertChannel):
    """Generic webhook alert channel."""

    def __init__(self, url: str, headers: dict | None = None, timeout: int = 5):
        self._url = url
        self._headers = headers or {"Content-Type": "application/json"}
        self._timeout = timeout
        self._breaker = get_breaker("webhook")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "webhook"

    async def send(self, alert: Alert) -> bool:
        if not self._url:
            logger.warn("webhook.no_url_configured")
            return False

        async def _do_send():
            client = self._client or httpx.AsyncClient(timeout=self._timeout)
            resp = await client.post(
                self._url,
                json=alert.to_dict(),
                headers=self._headers,
            )
            resp.raise_for_status()
            return True

        try:
            return await self._breaker.call(_do_send)
        except Exception as e:
            logger.error("webhook.send_failed", error=str(e), url=self._url)
            return False


class SlackChannel(AlertChannel):
    """Slack webhook alert channel."""

    def __init__(self, webhook_url: str, channel: str = "#alerts", username: str = "Anomaly Detector"):
        self._webhook_url = webhook_url
        self._channel = channel
        self._username = username
        self._breaker = get_breaker("slack")

    @property
    def name(self) -> str:
        return "slack"

    def _format_slack_message(self, alert: Alert) -> dict:
        """Format alert as Slack Block Kit message."""
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}
        color = {"info": "#36a3eb", "warning": "#f5a623", "critical": "#e74c3c"}

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji.get(alert.severity.value, '⚠️')} Anomaly Alert — {alert.severity.value.upper()}",
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Server:* `{alert.server_id}`"},
                    {"type": "mrkdwn", "text": f"*Metric:* `{alert.metric_name}`"},
                    {"type": "mrkdwn", "text": f"*Score:* `{alert.anomaly_score:.2f}`"},
                    {"type": "mrkdwn", "text": f"*Predicted:* {'Yes' if alert.is_predicted else 'No'}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Message:* {alert.message}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Suggested Action:* {alert.suggested_action}"},
            },
        ]

        if alert.dashboard_url:
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View Dashboard"},
                        "url": alert.dashboard_url,
                    }
                ],
            })

        return {
            "channel": self._channel,
            "username": self._username,
            "attachments": [{"color": color.get(alert.severity.value, "#999"), "blocks": blocks}],
        }

    async def send(self, alert: Alert) -> bool:
        if not self._webhook_url:
            return False

        async def _do_send():
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    self._webhook_url,
                    json=self._format_slack_message(alert),
                )
                resp.raise_for_status()
                return True

        try:
            return await self._breaker.call(_do_send)
        except Exception as e:
            logger.error("slack.send_failed", error=str(e))
            return False


class PagerDutyChannel(AlertChannel):
    """PagerDuty Events API v2 alert channel."""

    def __init__(self, routing_key: str, severity_map: dict | None = None):
        self._routing_key = routing_key
        self._severity_map = severity_map or {
            "info": "info", "warning": "warning", "critical": "critical",
        }
        self._breaker = get_breaker("pagerduty")

    @property
    def name(self) -> str:
        return "pagerduty"

    def _format_pd_event(self, alert: Alert) -> dict:
        severity = self._severity_map.get(alert.severity.value, "warning")
        return {
            "routing_key": self._routing_key,
            "event_action": "trigger",
            "dedup_key": f"anomaly-{alert.server_id}-{alert.metric_name}",
            "payload": {
                "summary": alert.message,
                "severity": severity,
                "source": alert.server_id,
                "component": alert.metric_name,
                "group": "infrastructure",
                "class": "anomaly_detection",
                "custom_details": alert.model_details,
                "timestamp": alert.timestamp.isoformat(),
            },
        }

    async def send(self, alert: Alert) -> bool:
        if not self._routing_key:
            return False

        async def _do_send():
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=self._format_pd_event(alert),
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                return True

        try:
            return await self._breaker.call(_do_send)
        except Exception as e:
            logger.error("pagerduty.send_failed", error=str(e))
            return False


class EmailChannel(AlertChannel):
    """Email alert channel (SMTP)."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender: str,
        recipients: list[str],
        smtp_tls: bool = True,
        smtp_user: str = "",
        smtp_pass: str = "",
    ):
        self._host = smtp_host
        self._port = smtp_port
        self._sender = sender
        self._recipients = recipients
        self._tls = smtp_tls
        self._user = smtp_user
        self._pass = smtp_pass

    @property
    def name(self) -> str:
        return "email"

    async def send(self, alert: Alert) -> bool:
        """Send alert via email (runs in thread pool to avoid blocking)."""
        from email.mime.text import MIMEText

        import aiosmtplib

        subject = f"[{alert.severity.value.upper()}] Anomaly: {alert.metric_name} on {alert.server_id}"
        body = alert.to_json()

        msg = MIMEText(body, "html")
        msg["Subject"] = subject
        msg["From"] = self._sender
        msg["To"] = ", ".join(self._recipients)

        try:
            await aiosmtplib.send(
                msg,
                hostname=self._host,
                port=self._port,
                username=self._user or None,
                password=self._pass or None,
                use_tls=self._tls,
            )
            return True
        except Exception as e:
            logger.error("email.send_failed", error=str(e))
            return False


class AlertDispatcher:
    """Multi-channel alert dispatcher with priority routing."""

    def __init__(self, config: dict | None = None):
        cfg = (config or get_config())["alerting"]["channels"]
        self._channels: list[AlertChannel] = []

        if cfg.get("webhook", {}).get("enabled"):
            self._channels.append(WebhookChannel(
                url=cfg["webhook"]["url"],
                headers=cfg["webhook"].get("headers", {}),
                timeout=cfg["webhook"].get("timeout", 5),
            ))

        if cfg.get("slack", {}).get("enabled"):
            self._channels.append(SlackChannel(
                webhook_url=cfg["slack"]["webhook_url"],
                channel=cfg["slack"].get("channel", "#alerts"),
                username=cfg["slack"].get("username", "Anomaly Detector"),
            ))

        if cfg.get("pagerduty", {}).get("enabled"):
            self._channels.append(PagerDutyChannel(
                routing_key=cfg["pagerduty"]["routing_key"],
                severity_map=cfg["pagerduty"].get("severity_map"),
            ))

        if cfg.get("email", {}).get("enabled"):
            self._channels.append(EmailChannel(
                smtp_host=cfg["email"]["smtp_host"],
                smtp_port=cfg["email"]["smtp_port"],
                sender=cfg["email"]["sender"],
                recipients=cfg["email"]["recipients"],
                smtp_tls=cfg["email"].get("smtp_tls", True),
            ))

    def add_channel(self, channel: AlertChannel) -> None:
        self._channels.append(channel)

    async def dispatch(self, alert: Alert) -> dict[str, bool]:
        """Dispatch alert to all configured channels concurrently.

        Returns:
            Dict of channel_name -> success
        """
        if not self._channels:
            logger.warn("dispatcher.no_channels")
            return {}

        results = await asyncio.gather(
            *[ch.send(alert) for ch in self._channels],
            return_exceptions=True,
        )

        channel_results = {}
        for channel, result in zip(self._channels, results, strict=False):
            if isinstance(result, Exception):
                channel_results[channel.name] = False
                logger.error(
                    "dispatcher.channel_error",
                    channel=channel.name,
                    error=str(result),
                )
            else:
                channel_results[channel.name] = result

        logger.info(
            "dispatcher.dispatch_complete",
            alert_id=alert.alert_id,
            channels=channel_results,
        )
        return channel_results

    @property
    def channel_names(self) -> list[str]:
        return [ch.name for ch in self._channels]
