"""Payment provider abstraction (YooKassa-shaped, vendor-neutral)."""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel

PaymentStatus = Literal["pending", "paid", "canceled"]


class Payment(BaseModel):
    """A payment record gating a paid action."""

    id: str
    amount: float
    currency: str = "RUB"
    status: PaymentStatus = "pending"
    confirmation_url: str | None = None


class PaymentProvider(Protocol):
    """Create, confirm and look up payments."""

    def create(self, amount: float, currency: str = "RUB") -> Payment: ...

    def confirm(self, payment_id: str) -> Payment: ...

    def get(self, payment_id: str) -> Payment | None: ...
