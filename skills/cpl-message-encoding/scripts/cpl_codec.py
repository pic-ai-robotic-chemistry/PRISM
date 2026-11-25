"""Deterministic codec for spatially multiplexed CPL messages.

Each cell carries a colour/Morse token and one quaternary ASCII digit.
The configuration library maps all twelve colour/intensity pairs to measured
red-phase rows.
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Iterable


PROTOCOL_NAME = "cpl-spatial-multiplex"

COLORS = ("B", "G", "R")
RADIX = 4
ASCII_CODE_WIDTH = 4
MORSE_BLOCK_WIDTH = 4
IS_RED_REQUIRED = 1
DEFAULT_ROW_PLAINTEXT_CHARACTERS = 4
FIXED_ROW_CONFIGURATION_COUNT = 2 * ASCII_CODE_WIDTH
DEFAULT_FILLER_COLOR = "B"
DEFAULT_FILLER_DIGIT = 1
MESSAGE_LIBRARY_TYPE = "cpl_message_configuration_library"
MODEL_PREDICTION_SOURCE = "inverse_design_prediction"

INTENSITY_INTERVAL_NAMES = {
    0: "outer_negative",
    1: "inner_negative",
    2: "inner_positive",
    3: "outer_positive",
}
INTENSITY_INTERVAL_LABELS = {
    0: "< -5000 mdeg",
    1: "[-5000, 0) mdeg",
    2: "[0, 5000] mdeg",
    3: "> 5000 mdeg",
}

MORSE_BY_LETTER = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
}
LETTER_BY_MORSE = {code: letter for letter, code in MORSE_BY_LETTER.items()}

MORSE_TOKEN_BY_COLOR = {"B": ".", "G": "-", "R": "separator_or_space"}
COLOR_BY_MORSE_MARK = {".": "B", "-": "G"}

# Stable IDs are colour-major, then quaternary digit.
DESIGN_ID_BY_PAIR = {
    (color, digit): f"D{color_index * RADIX + digit + 1:03d}"
    for color_index, color in enumerate(COLORS)
    for digit in range(RADIX)
}
PAIR_BY_DESIGN_ID = {design_id: pair for pair, design_id in DESIGN_ID_BY_PAIR.items()}


class ProtocolError(ValueError):
    """Raised when a library or message violates the physical protocol."""


def normalize_message(message: str) -> str:
    """Uppercase A-Z text and canonicalize runs of ASCII spaces."""
    if not isinstance(message, str):
        raise ProtocolError("Message must be a string.")
    upper = message.upper()
    unsupported = sorted({char for char in upper if char != " " and not ("A" <= char <= "Z")})
    if unsupported:
        rendered = ", ".join(repr(char) for char in unsupported)
        raise ProtocolError(
            f"Unsupported plaintext character(s): {rendered}. "
            "The protocol accepts only English A-Z and ASCII spaces."
        )
    normalized = " ".join(upper.split())
    if not normalized:
        raise ProtocolError("Message must contain at least one A-Z character.")
    return normalized


def integer_to_base_digits(value: int, radix: int) -> list[int]:
    """Convert an integer; callers apply any protocol-specific fixed-width padding."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError("ASCII value must be a non-negative integer.")
    if radix < 2 or radix > 10:
        raise ProtocolError("Radix must be between 2 and 10.")
    if value == 0:
        return [0]
    digits: list[int] = []
    while value:
        value, digit = divmod(value, radix)
        digits.append(digit)
    return list(reversed(digits))


def ascii_digits(character: str, radix: int = RADIX) -> list[int]:
    """Encode one supported character as fixed-width, four-digit base-4 ASCII."""
    if len(character) != 1 or (character != " " and not ("A" <= character <= "Z")):
        raise ProtocolError(f"No protocol ASCII mapping for {character!r}.")
    if radix != RADIX:
        raise ProtocolError("The protocol supports only radix 4.")
    digits = integer_to_base_digits(ord(character), radix)
    if len(digits) > ASCII_CODE_WIDTH:
        raise ProtocolError(f"ASCII value for {character!r} exceeds four base-4 digits.")
    return [0] * (ASCII_CODE_WIDTH - len(digits)) + digits


def classify_intensity(value: Any, *, location: str) -> int:
    """Map a signed peak intensity to the exact four requested intervals."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{location}: peak_cd_intensity must be numeric.")
    intensity = float(value)
    if not isfinite(intensity):
        raise ProtocolError(f"{location}: peak_cd_intensity must be finite.")
    if intensity < -5000.0:
        return 0
    if intensity < 0.0:
        return 1
    if intensity <= 5000.0:
        return 2
    return 3


def protocol_metadata() -> dict[str, Any]:
    """Build the authoritative, self-describing protocol object."""
    return {
        "name": PROTOCOL_NAME,
        "is_red_required": IS_RED_REQUIRED,
        "plaintext_alphabet": "A-Z and ASCII space; uppercase; ASCII-space runs collapsed and stripped",
        "row_plaintext_characters": DEFAULT_ROW_PLAINTEXT_CHARACTERS,
        "alternation": {
            "zero_based_even_plaintext_positions": "morse_from_colour",
            "zero_based_odd_plaintext_positions": "ascii_from_intensity",
            "spaces_consume_plaintext_positions": True,
        },
        "row_framing": {
            "payload_character_count_field": "rows[].payload_character_count",
            "full_row_plaintext_characters": DEFAULT_ROW_PLAINTEXT_CHARACTERS,
            "full_row_morse_characters": 2,
            "full_row_ascii_characters": 2,
            "fixed_configuration_count": FIXED_ROW_CONFIGURATION_COUNT,
            "every_row_has_fixed_configuration_count": True,
            "final_partial_row_uses_explicit_count": True,
        },
        "morse": {
            "colour_tokens": MORSE_TOKEN_BY_COLOR,
            "codes": MORSE_BY_LETTER,
            "short_code_terminator_colour": "R",
            "four_mark_code_ends_at_subblock_boundary": True,
            "space_code_colours": ["R"],
            "characters_per_full_row": 2,
            "colour_slots_per_character": MORSE_BLOCK_WIDTH,
            "fixed_colour_slots_per_row": FIXED_ROW_CONFIGURATION_COUNT,
            "read_rule": (
                "split the row into two four-slot subblocks and decode one declared "
                "Morse character per subblock; the first R terminates a short code, "
                "while no R means a four-mark code"
            ),
            "filler_colour_after_payload": DEFAULT_FILLER_COLOR,
        },
        "ascii": {
            "radix": RADIX,
            "code_width_digits": ASCII_CODE_WIDTH,
            "representation": (
                "fixed-width four-digit unsigned standard ASCII in base 4, "
                "zero-padded, most-significant digit first"
            ),
            "codes": {char: ascii_digits(char, RADIX) for char in (" ", *MORSE_BY_LETTER)},
            "digit_intervals": {
                "0": {
                    "name": INTENSITY_INTERVAL_NAMES[0],
                    "rule": "peak_cd_intensity < -5000",
                    "label": INTENSITY_INTERVAL_LABELS[0],
                },
                "1": {
                    "name": INTENSITY_INTERVAL_NAMES[1],
                    "rule": "-5000 <= peak_cd_intensity < 0",
                    "label": INTENSITY_INTERVAL_LABELS[1],
                },
                "2": {
                    "name": INTENSITY_INTERVAL_NAMES[2],
                    "rule": "0 <= peak_cd_intensity <= 5000",
                    "label": INTENSITY_INTERVAL_LABELS[2],
                },
                "3": {
                    "name": INTENSITY_INTERVAL_NAMES[3],
                    "rule": "peak_cd_intensity > 5000",
                    "label": INTENSITY_INTERVAL_LABELS[3],
                },
            },
            "read_rule": (
                "read exactly rows[].ascii_character_count * code_width_digits digits"
            ),
            "full_row_meaningful_digits": 2 * ASCII_CODE_WIDTH,
            "characters_per_full_row": 2,
            "fixed_intensity_slots_per_row": FIXED_ROW_CONFIGURATION_COUNT,
            "filler_digit_after_payload": DEFAULT_FILLER_DIGIT,
        },
        "spatial_multiplexing": {
            "one_cell": "one colour token and one quaternary intensity digit in the same device",
            "fixed_configurations_per_row": FIXED_ROW_CONFIGURATION_COUNT,
            "configuration_slots": len(DESIGN_ID_BY_PAIR),
            "configuration_pairs": "3 colours x 4 intensity digits",
        },
    }


def _validate_protocol(protocol: Any) -> int:
    if not isinstance(protocol, dict) or protocol.get("name") != PROTOCOL_NAME:
        raise ProtocolError(f"Protocol must be {PROTOCOL_NAME!r}.")
    row_size = protocol.get("row_plaintext_characters")
    if (
        isinstance(row_size, bool)
        or not isinstance(row_size, int)
        or row_size != DEFAULT_ROW_PLAINTEXT_CHARACTERS
    ):
        raise ProtocolError("protocol.row_plaintext_characters must be exactly 4.")
    expected = protocol_metadata()
    if protocol != expected:
        raise ProtocolError("Protocol metadata differs from the canonical definition.")
    return row_size


def _read_observed_pair(
    configuration: dict[str, Any], expected_pair: tuple[str, int], *, location: str
) -> tuple[str, int]:
    material = configuration.get("material_parameters")
    observables = configuration.get("observables")
    if not isinstance(material, dict) or not isinstance(observables, dict):
        raise ProtocolError(f"{location}: missing material_parameters or observables.")
    if material.get("is_red") != IS_RED_REQUIRED:
        raise ProtocolError(f"{location}: every configuration must use red phase is_red=1.")
    if material.get("color") != expected_pair[0]:
        raise ProtocolError(f"{location}: material colour must match the intended pair.")
    observed_colour = observables.get("colour_symbol")
    if observed_colour not in COLORS:
        raise ProtocolError(f"{location}: observable colour must be one of {COLORS}.")
    observed_digit = classify_intensity(observables.get("peak_cd_intensity"), location=location)
    observed_pair = (observed_colour, observed_digit)
    if observed_pair != expected_pair:
        raise ProtocolError(
            f"{location}: colour/intensity decodes as {observed_pair}, "
            f"not intended pair {expected_pair}."
        )
    return observed_pair


def validate_library(
    library: Any, *, required_pairs: set[tuple[str, int]] | None = None
) -> dict[tuple[str, int], dict[str, Any]]:
    """Validate every supplied configuration and require the requested pairs."""
    if not isinstance(library, dict):
        raise ProtocolError("Configuration library must be a JSON object.")
    _validate_protocol(library.get("protocol"))
    catalog = library.get("design_catalog")
    if not isinstance(catalog, dict):
        raise ProtocolError("Configuration library design_catalog must be an object.")
    stable_ids = set(PAIR_BY_DESIGN_ID)
    required_ids = stable_ids if required_pairs is None else {DESIGN_ID_BY_PAIR[pair] for pair in required_pairs}
    missing = sorted(required_ids - set(catalog))
    extra = sorted(set(catalog) - stable_ids)
    if missing or extra:
        raise ProtocolError(f"Configuration library does not cover the required IDs; missing={missing}, extra={extra}.")

    by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for design_id, design in catalog.items():
        if not isinstance(design, dict):
            raise ProtocolError(f"Library {design_id} must be an object.")
        expected_pair = PAIR_BY_DESIGN_ID[design_id]
        if design.get("colour_symbol") != expected_pair[0] or design.get("intensity_digit") != expected_pair[1]:
            raise ProtocolError(f"Library {design_id} does not match stable pair {expected_pair}.")
        _read_observed_pair(design, expected_pair, location=f"Library {design_id}")
        by_pair[expected_pair] = design
    return by_pair


def encode_morse_character(character: str) -> list[str]:
    """Encode one Morse character before fixed-subblock filler is applied."""
    if character == " ":
        return ["R"]
    colours = [COLOR_BY_MORSE_MARK[mark] for mark in MORSE_BY_LETTER[character]]
    if len(colours) < MORSE_BLOCK_WIDTH:
        colours.append("R")
    return colours


def encode_morse_channel(characters: Iterable[str]) -> list[str]:
    """Encode at most two Morse characters into independent four-slot subblocks."""
    row_characters = list(characters)
    if len(row_characters) > 2:
        raise ProtocolError("The protocol permits at most two Morse characters per row.")
    colours: list[str] = []
    for character in row_characters:
        payload = encode_morse_character(character)
        colours.extend(
            payload + [DEFAULT_FILLER_COLOR] * (MORSE_BLOCK_WIDTH - len(payload))
        )
    return colours


def encode_ascii_channel(characters: Iterable[str], radix: int = RADIX) -> list[int]:
    digits: list[int] = []
    for character in characters:
        digits.extend(ascii_digits(character, radix))
    return digits


def plan_symbol_rows(normalized_message: str) -> list[dict[str, Any]]:
    """Frame both channels before assigning material parameters."""
    if normalize_message(normalized_message) != normalized_message:
        raise ProtocolError("Row planning requires canonical normalized plaintext.")
    row_size = DEFAULT_ROW_PLAINTEXT_CHARACTERS
    rows: list[dict[str, Any]] = []
    for row_index, start in enumerate(range(0, len(normalized_message), row_size)):
        plaintext = normalized_message[start : start + row_size]
        morse_characters = list(plaintext[0::2])
        ascii_characters = list(plaintext[1::2])
        morse_colour_payloads = [
            encode_morse_character(character) for character in morse_characters
        ]
        colour_stream = encode_morse_channel(morse_characters)
        intensity_stream = encode_ascii_channel(ascii_characters, RADIX)
        if (
            len(colour_stream) > FIXED_ROW_CONFIGURATION_COUNT
            or len(intensity_stream) > FIXED_ROW_CONFIGURATION_COUNT
        ):
            raise ProtocolError(
                f"Row {row_index}: channel payload exceeds eight fixed configuration slots."
            )
        colour_slots = colour_stream + [DEFAULT_FILLER_COLOR] * (
            FIXED_ROW_CONFIGURATION_COUNT - len(colour_stream)
        )
        intensity_slots = intensity_stream + [DEFAULT_FILLER_DIGIT] * (
            FIXED_ROW_CONFIGURATION_COUNT - len(intensity_stream)
        )
        rows.append(
            {
                "row_index": row_index,
                "payload_character_count": len(plaintext),
                "morse_character_count": len(morse_characters),
                "ascii_character_count": len(ascii_characters),
                "plaintext": plaintext,
                "morse_characters": morse_characters,
                "ascii_characters": ascii_characters,
                "morse_colour_payloads": morse_colour_payloads,
                "colour_stream": colour_stream,
                "intensity_stream": intensity_stream,
                "colour_slots": colour_slots,
                "intensity_slots": intensity_slots,
            }
        )
    return rows


def required_configuration_pairs(rows: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {pair for row in rows for pair in zip(row["colour_slots"], row["intensity_slots"])}


def attach_configurations(rows: list[dict[str, Any]], library: dict[str, Any]) -> list[dict[str, Any]]:
    required_pairs = None
    if isinstance(library, dict) and library.get("library_type") == MESSAGE_LIBRARY_TYPE:
        required_pairs = required_configuration_pairs(rows)
    by_pair = validate_library(library, required_pairs=required_pairs)
    plans = []
    for row in rows:
        cells = []
        for column, pair in enumerate(zip(row["colour_slots"], row["intensity_slots"])):
            colour, digit = pair
            design = by_pair[pair]
            cells.append({
                "column": column,
                "design_id": DESIGN_ID_BY_PAIR[pair],
                "intended_pair": {"colour_symbol": colour, "intensity_digit": digit},
                "material_parameters": dict(design["material_parameters"]),
                "observables": dict(design["observables"]),
            })
        plans.append({**row, "cells": cells})
    return plans


def build_row_plans(normalized_message: str, library: dict[str, Any]) -> list[dict[str, Any]]:
    """Create eight-device rows from a message and a configuration library."""
    return attach_configurations(plan_symbol_rows(normalized_message), library)


def decode_morse_channel(colours: list[str], payload_character_count: int) -> list[str]:
    """Decode up to two Morse characters from independent four-slot subblocks."""
    if payload_character_count not in (0, 1, 2):
        raise ProtocolError("Morse payload character count must be 0, 1, or 2.")
    if len(colours) != FIXED_ROW_CONFIGURATION_COUNT:
        raise ProtocolError(
            f"Morse row must contain exactly {FIXED_ROW_CONFIGURATION_COUNT} colour slots."
        )
    return [
        _decode_morse_block(
            colours[
                block_index * MORSE_BLOCK_WIDTH : (block_index + 1) * MORSE_BLOCK_WIDTH
            ]
        )
        for block_index in range(payload_character_count)
    ]


def _decode_morse_block(colours: list[str]) -> str:
    """Decode one four-slot Morse subblock."""
    if len(colours) != MORSE_BLOCK_WIDTH:
        raise ProtocolError(f"Morse subblock must contain {MORSE_BLOCK_WIDTH} colours.")
    buffer = ""
    for colour in colours:
        if colour == "B":
            buffer += "."
        elif colour == "G":
            buffer += "-"
        elif colour == "R":
            if not buffer:
                return " "
            letter = LETTER_BY_MORSE.get(buffer)
            if letter is None:
                raise ProtocolError(f"Invalid Morse code {buffer!r}.")
            return letter
        else:
            raise ProtocolError(f"Unsupported Morse colour {colour!r}.")
    letter = LETTER_BY_MORSE.get(buffer)
    if letter is None:
        raise ProtocolError(f"Invalid four-mark Morse code {buffer!r}.")
    return letter


ASCII_CYCLE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "


def decode_ascii_channel(
    digits: list[int],
    *,
    radix: int,
    payload_character_count: int,
) -> list[str]:
    """Decode exactly N fixed-width base-4 codewords, then ignore filler.

    Values 32 (space) and 65-90 (A-Z) keep their standard meaning.  Any other
    value cycles into the 27-symbol payload alphabet by modulo 27, so decoding
    never fails on out-of-alphabet codewords and never emits a character
    outside A-Z plus space.
    """
    if radix != RADIX:
        raise ProtocolError("The protocol supports only radix 4.")
    if payload_character_count < 0:
        raise ProtocolError("ASCII payload character count must be non-negative.")
    meaningful_length = payload_character_count * ASCII_CODE_WIDTH
    if len(digits) < meaningful_length:
        raise ProtocolError(
            f"ASCII row has {len(digits)} digits; {meaningful_length} are required."
        )
    characters: list[str] = []
    for start in range(0, meaningful_length, ASCII_CODE_WIDTH):
        code = digits[start : start + ASCII_CODE_WIDTH]
        if any(digit not in range(RADIX) for digit in code):
            raise ProtocolError(f"Invalid base-4 ASCII codeword {code}.")
        value = 0
        for digit in code:
            value = value * RADIX + digit
        if value == 32:
            characters.append(" ")
        elif 65 <= value <= 90:
            characters.append(chr(value))
        else:
            characters.append(ASCII_CYCLE_ALPHABET[value % len(ASCII_CYCLE_ALPHABET)])
    return characters


def _read_cell(
    cell: Any,
    *,
    row_index: int,
    column: int,
) -> tuple[str, int]:
    location = f"Row {row_index}, column {column}"
    if not isinstance(cell, dict) or cell.get("column") != column:
        raise ProtocolError(f"{location}: cell is malformed or column index is not contiguous.")
    intended = cell.get("intended_pair")
    if not isinstance(intended, dict):
        raise ProtocolError(f"{location}: missing intended_pair framing metadata.")
    colour = intended.get("colour_symbol")
    digit = intended.get("intensity_digit")
    if colour not in COLORS or type(digit) is not int or digit not in range(RADIX):
        raise ProtocolError(f"{location}: intended colour/digit pair is invalid.")
    expected_id = DESIGN_ID_BY_PAIR[(colour, digit)]
    if cell.get("design_id") != expected_id:
        raise ProtocolError(f"{location}: design_id does not match intended pair.")

    return _read_observed_pair(cell, (colour, digit), location=location)


def decode_document(
    document: Any,
    expected: str | None = None,
) -> dict[str, Any]:
    """Decode ordered colour and intensity observations."""
    if not isinstance(document, dict):
        raise ProtocolError("Top-level JSON value must be an object.")
    row_size = _validate_protocol(document.get("protocol"))
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ProtocolError("rows must be a non-empty array.")

    decoded_rows: list[dict[str, Any]] = []
    all_characters: list[str] = []
    configuration_count = 0
    predicted_cell_count = 0
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("row_index") != row_index:
            raise ProtocolError(f"Row {row_index}: row index is not contiguous.")
        payload_count = row.get("payload_character_count")
        morse_count = row.get("morse_character_count")
        ascii_count = row.get("ascii_character_count")
        if (
            isinstance(payload_count, bool)
            or not isinstance(payload_count, int)
            or payload_count < 1
            or payload_count > row_size
        ):
            raise ProtocolError(
                f"Row {row_index}: payload_character_count must be in [1, {row_size}]."
            )
        if row_index < len(rows) - 1 and payload_count != row_size:
            raise ProtocolError(
                f"Row {row_index}: every non-final row must frame {row_size} plaintext characters."
            )
        expected_morse_count = (payload_count + 1) // 2
        expected_ascii_count = payload_count // 2
        if (
            isinstance(morse_count, bool)
            or not isinstance(morse_count, int)
            or isinstance(ascii_count, bool)
            or not isinstance(ascii_count, int)
            or morse_count != expected_morse_count
            or ascii_count != expected_ascii_count
        ):
            raise ProtocolError(
                f"Row {row_index}: channel counts must be Morse={expected_morse_count}, "
                f"ASCII={expected_ascii_count} for payload count {payload_count}."
            )
        cells = row.get("cells")
        if row.get("cell_count") != FIXED_ROW_CONFIGURATION_COUNT:
            raise ProtocolError(
                f"Row {row_index}: cell_count must be {FIXED_ROW_CONFIGURATION_COUNT}."
            )
        if not isinstance(cells, list) or len(cells) != FIXED_ROW_CONFIGURATION_COUNT:
            raise ProtocolError(
                f"Row {row_index}: cells must contain exactly "
                f"{FIXED_ROW_CONFIGURATION_COUNT} configurations."
            )
        colours: list[str] = []
        digits: list[int] = []
        for column, cell in enumerate(cells):
            colour, digit = _read_cell(cell, row_index=row_index, column=column)
            colours.append(colour)
            digits.append(digit)
            predicted_cell_count += cell["observables"].get("source") == MODEL_PREDICTION_SOURCE
        configuration_count += len(cells)

        morse_characters = decode_morse_channel(colours, morse_count)
        ascii_characters = decode_ascii_channel(
            digits, radix=RADIX, payload_character_count=ascii_count
        )
        merged: list[str] = []
        for position in range(max(len(morse_characters), len(ascii_characters))):
            if position < len(morse_characters):
                merged.append(morse_characters[position])
            if position < len(ascii_characters):
                merged.append(ascii_characters[position])
        if len(merged) != payload_count:
            raise ProtocolError(
                f"Row {row_index}: decoded {len(merged)} characters; "
                f"framing declares {payload_count}."
            )
        all_characters.extend(merged)
        decoded_rows.append(
            {
                "row_index": row_index,
                "payload_character_count": payload_count,
                "morse_characters": "".join(morse_characters),
                "ascii_characters": "".join(ascii_characters),
                "interleaved": "".join(merged),
                "configuration_count": len(cells),
            }
        )

    decoded = "".join(all_characters)
    normalized_expected = normalize_message(expected) if expected is not None else None
    matches = decoded == normalized_expected if normalized_expected is not None else None
    return {
        "success": matches is not False,
        "decoded_message": decoded,
        "configuration_count": configuration_count,
        "predicted_cell_count": predicted_cell_count,
        "row_count": len(rows),
        "rows": decoded_rows,
        "comparison_performed": expected is not None,
        "expected_message": normalized_expected,
        "matches_expected": matches,
    }
