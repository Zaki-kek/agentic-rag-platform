"""In-memory stub payment provider.

Models the create -> redirect -> webhook-confirm lifecycle of a real gateway
(e.g. YooKassa) without any credentials, so paid flows are demoable and testable.
Swap in a real provider behind the same PaymentProvider protocol.
"""

from __future__ import annotations

import uuid

from app.core import AppError
from app.payments.base import Payment


class PaymentNotFoundError(AppError):
    status_code = 404


class StubPaymentProvider:
    """Deterministic, in-memory payment provider for demos and tests."""

    def __init__(self) -> None:
        self._payments: dict[str, Payment] = {}

    def create(self, amount: float, currency: str = "RUB") -> Payment:
        payment_id = uuid.uuid4().hex
        payment = Payment(
            id=payment_id,
            amount=amount,
            currency=currency,
            status="pending",
            confirmation_url=f"https://pay.example/checkout/{payment_id}",
        )
        self._payments[payment_id] = payment
        return payment

    def confirm(self, payment_id: str) -> Payment:
        payment = self.get(payment_id)
        if payment is None:
            raise PaymentNotFoundError(f"payment '{payment_id}' not found")
        payment.status = "paid"
        return payment

    def get(self, payment_id: str) -> Payment | None:
        return self._payments.get(payment_id)
