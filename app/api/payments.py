"""/payments routes: create, confirm (webhook-style) and look up payments."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.payments.base import Payment

router = APIRouter(tags=["payments"])


class PaymentRequest(BaseModel):
    amount: float = Field(..., gt=0)
    currency: str = "RUB"


@router.post("/payments", response_model=Payment)
async def create_payment(payload: PaymentRequest, request: Request) -> Payment:
    return request.app.state.payment_provider.create(payload.amount, payload.currency)


@router.post("/payments/{payment_id}/confirm", response_model=Payment)
async def confirm_payment(payment_id: str, request: Request) -> Payment:
    # raises PaymentNotFoundError (handled as 404) if missing
    return request.app.state.payment_provider.confirm(payment_id)


@router.get("/payments/{payment_id}", response_model=Payment)
async def get_payment(payment_id: str, request: Request) -> Payment:
    payment = request.app.state.payment_provider.get(payment_id)
    if payment is None:
        raise HTTPException(status_code=404, detail="payment not found")
    return payment
