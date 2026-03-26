from __future__ import annotations

from fractions import Fraction
from typing import Iterable


def parse_ratio(value: str | int | float | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, float):
        return Fraction(str(value))
    text = value.strip()
    if "/" in text:
        numerator, denominator = text.split("/", maxsplit=1)
        return Fraction(int(numerator), int(denominator))
    return Fraction(text)


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def format_ratio(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_pretty_number(
    value: float,
    *,
    tolerance: float = 1e-9,
    max_denominator: int = 10_000,
    decimal_places: int = 12,
) -> str:
    normalized = 0.0 if abs(value) <= tolerance else float(value)
    candidate = Fraction(normalized).limit_denominator(max_denominator)
    if abs(float(candidate) - normalized) <= tolerance:
        return format_ratio(candidate)

    text = f"{normalized:.{decimal_places}f}".rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", ""}:
        return "0"
    return text


def format_sorted_numeric_values(
    values: Iterable[float],
    *,
    tolerance: float = 1e-9,
    max_denominator: int = 10_000,
    decimal_places: int = 12,
) -> str:
    normalized_values = [0.0 if abs(value) <= tolerance else float(value) for value in values]
    normalized_values.sort(reverse=True)
    rendered = [
        format_pretty_number(
            value,
            tolerance=tolerance,
            max_denominator=max_denominator,
            decimal_places=decimal_places,
        )
        for value in normalized_values
    ]
    return "[" + ", ".join(rendered) + "]"


def format_decimal_number(
    value: float,
    *,
    tolerance: float = 1e-9,
    decimal_places: int = 12,
) -> str:
    normalized = 0.0 if abs(value) <= tolerance else float(value)
    text = f"{normalized:.{decimal_places}f}".rstrip("0").rstrip(".")
    if text in {"-0", "-0.0", ""}:
        return "0"
    return text


def format_sorted_decimal_values(
    values: Iterable[float],
    *,
    tolerance: float = 1e-9,
    decimal_places: int = 12,
) -> str:
    normalized_values = [0.0 if abs(value) <= tolerance else float(value) for value in values]
    normalized_values.sort(reverse=True)
    rendered = [
        format_decimal_number(
            value,
            tolerance=tolerance,
            decimal_places=decimal_places,
        )
        for value in normalized_values
    ]
    return "[" + ", ".join(rendered) + "]"


def format_scaled_rational_values(
    values: Iterable[float],
    *,
    tolerance: float = 1e-6,
    max_denominator: int = 1_000,
) -> str:
    normalized_values = [0.0 if abs(value) <= tolerance else float(value) for value in values]
    normalized_values.sort(reverse=True)

    for denominator in range(1, max_denominator + 1):
        scaled_numerators = [int(round(value * denominator)) for value in normalized_values]
        if all(
            abs(value - scaled_numerator / denominator) <= tolerance
            for value, scaled_numerator in zip(normalized_values, scaled_numerators)
        ):
            return "[" + ", ".join(str(numerator) for numerator in scaled_numerators) + "]"

    return format_sorted_decimal_values(
            normalized_values,
            tolerance=tolerance,
            decimal_places=12,
        )
