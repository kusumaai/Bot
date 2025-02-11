from decimal import Decimal, ROUND_HALF_UP

PRICE_PRECISION = Decimal('0.00000001')
SIZE_PRECISION = Decimal('0.00000001')
PNL_PRECISION = Decimal('0.00000001')

def normalize_decimal(value: Decimal, precision: Decimal) -> Decimal:
    return Decimal(str(value)).quantize(precision, rounding=ROUND_HALF_UP) 