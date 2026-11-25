"""Deterministic tests for the eight-configuration CPL protocol."""

from __future__ import annotations

import copy
import unittest

from cpl_codec import (
    DESIGN_ID_BY_PAIR,
    FIXED_ROW_CONFIGURATION_COUNT,
    MESSAGE_LIBRARY_TYPE,
    RADIX,
    ProtocolError,
    ascii_digits,
    build_row_plans,
    classify_intensity,
    decode_ascii_channel,
    decode_document,
    normalize_message,
    plan_symbol_rows,
    protocol_metadata,
    required_configuration_pairs,
    validate_library,
)
from create_config_library import (
    best_phase_proxy,
    build_catalog,
)
from encode_cpl_message import render_configuration_sequence_markdown


FAKE_INTENSITY = {0: -8000.0, 1: -2500.0, 2: 2500.0, 3: 8000.0}


def sample_peak_rows() -> list[dict]:
    return [
        {
            "csv_line": line,
            "color": color,
            "is_red": True,
            "gray": 5.0,
            "thickness": 48.0,
            "stretch": 67.0 + digit,
            "peak_wavelength": {"B": 475.0, "G": 550.0, "R": 600.0}[color],
            "peak_cd_intensity": FAKE_INTENSITY[digit],
            "intensity_digit": digit,
        }
        for line, (color, digit) in enumerate(DESIGN_ID_BY_PAIR, start=2)
    ]


def make_library() -> dict:
    return {
        "library_id": "test-library",
        "protocol": protocol_metadata(),
        "design_catalog": build_catalog(sample_peak_rows(), minimum_margin=500.0),
    }


def make_document(message: str) -> tuple[dict, list[dict], dict]:
    library = make_library()
    normalized = normalize_message(message)
    plans = build_row_plans(normalized, library)
    document = {
        "protocol": library["protocol"],
        "rows": [
            {
                "row_index": plan["row_index"],
                "payload_character_count": plan["payload_character_count"],
                "morse_character_count": plan["morse_character_count"],
                "ascii_character_count": plan["ascii_character_count"],
                "cell_count": len(plan["cells"]),
                "cells": copy.deepcopy(plan["cells"]),
            }
            for plan in plans
        ],
    }
    return document, plans, library


class CPLCodecTests(unittest.TestCase):
    def test_message_library_covers_every_used_pair_including_padding(self) -> None:
        required = required_configuration_pairs(plan_symbol_rows("E"))
        self.assertEqual(required, {("B", 1), ("R", 1)})
        library = make_library()
        library["library_type"] = MESSAGE_LIBRARY_TYPE
        library["design_catalog"] = {
            design_id: library["design_catalog"][design_id]
            for design_id in (DESIGN_ID_BY_PAIR[pair] for pair in required)
        }
        self.assertEqual(len(build_row_plans("E", library)[0]["cells"]), 8)
        with self.assertRaisesRegex(ProtocolError, "missing="):
            build_row_plans("ABCK", library)

    def test_all_twelve_configurations_use_the_same_format(self) -> None:
        pairs = validate_library(make_library())
        self.assertEqual(set(pairs), set(DESIGN_ID_BY_PAIR))
        self.assertEqual(len({frozenset(design) for design in pairs.values()}), 1)
        for pair, design in pairs.items():
            with self.subTest(pair=pair):
                color, digit = pair
                self.assertEqual(design["design_id"], DESIGN_ID_BY_PAIR[pair])
                self.assertEqual(design["material_parameters"]["color"], color)
                self.assertEqual(design["material_parameters"]["is_red"], 1)
                self.assertEqual(design["observables"]["colour_symbol"], color)
                self.assertEqual(design["observables"]["peak_cd_intensity"], FAKE_INTENSITY[digit])
                self.assertIsNone(design["selection"]["phase_contrast"])

    def test_every_pair_requires_a_measured_row(self) -> None:
        rows = sample_peak_rows()
        for index, row in enumerate(rows):
            with self.subTest(color=row["color"], digit=row["intensity_digit"]):
                message = f"color={row['color']}, digit={row['intensity_digit']}"
                with self.assertRaisesRegex(ProtocolError, message):
                    build_catalog(rows[:index] + rows[index + 1 :], minimum_margin=500.0)

    def test_all_twelve_configurations_round_trip(self) -> None:
        phrase = "ABCKAKKMAAAM"
        document, plans, _ = make_document(phrase)
        self.assertEqual(
            {cell["design_id"] for plan in plans for cell in plan["cells"]},
            set(DESIGN_ID_BY_PAIR.values()),
        )
        result = decode_document(document, expected=phrase)
        self.assertTrue(result["matches_expected"])
        self.assertEqual(result["configuration_count"], 24)

    def test_exact_interval_boundaries(self) -> None:
        cases = [
            (-5000.0001, 0),
            (-5000.0, 1),
            (-0.0001, 1),
            (0.0, 2),
            (5000.0, 2),
            (5000.0001, 3),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(classify_intensity(value, location="test"), expected)

    def test_ascii_is_fixed_width_base4_and_has_no_space_terminator(self) -> None:
        self.assertEqual(RADIX, 4)
        self.assertEqual(ascii_digits(" "), [0, 2, 0, 0])
        self.assertEqual(ascii_digits("A"), [1, 0, 0, 1])
        self.assertEqual(ascii_digits("Z"), [1, 1, 2, 2])
        self.assertTrue(all(len(ascii_digits(char)) == 4 for char in " ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertNotIn("space_code_digits", protocol_metadata()["ascii"])
        self.assertNotIn("channel_terminator", protocol_metadata()["ascii"])
        self.assertNotIn("channel_terminator", protocol_metadata()["morse"])
        self.assertEqual(protocol_metadata()["ascii"]["full_row_meaningful_digits"], 8)

    def test_out_of_alphabet_ascii_values_cycle_into_payload_alphabet(self) -> None:
        self.assertEqual(
            decode_ascii_channel([0, 2, 0, 0], radix=RADIX, payload_character_count=1),
            [" "],
        )
        self.assertEqual(
            decode_ascii_channel([1, 0, 0, 1], radix=RADIX, payload_character_count=1),
            ["A"],
        )
        self.assertEqual(
            decode_ascii_channel([1, 1, 2, 2], radix=RADIX, payload_character_count=1),
            ["Z"],
        )
        # 224 % 27 == 8 -> "I"; 142 % 27 == 7 -> "H"; 48 % 27 == 21 -> "V".
        for digits, expected in (
            ([3, 2, 0, 0], "I"),
            ([2, 0, 3, 2], "H"),
            ([0, 3, 0, 0], "V"),
        ):
            with self.subTest(digits=digits):
                self.assertEqual(
                    decode_ascii_channel(digits, radix=RADIX, payload_character_count=1),
                    [expected],
                )

    def test_precise_channels_and_round_trip(self) -> None:
        document, plans, _ = make_document("Precise")
        self.assertEqual([plan["plaintext"] for plan in plans], ["PREC", "ISE"])
        self.assertEqual(plans[0]["morse_characters"], ["P", "E"])
        self.assertEqual(plans[0]["ascii_characters"], ["R", "C"])
        self.assertEqual(
            plans[0]["colour_stream"],
            ["B", "G", "G", "B", "B", "R", "B", "B"],
        )
        self.assertEqual(plans[0]["intensity_stream"], [1, 1, 0, 2, 1, 0, 0, 3])
        self.assertEqual(
            [cell["design_id"] for cell in plans[0]["cells"]],
            ["D002", "D006", "D005", "D003", "D002", "D009", "D001", "D004"],
        )
        self.assertEqual(plans[0]["payload_character_count"], 4)
        self.assertEqual(plans[0]["morse_character_count"], 2)
        self.assertEqual(plans[0]["ascii_character_count"], 2)
        self.assertTrue(
            all(len(plan["cells"]) == FIXED_ROW_CONFIGURATION_COUNT for plan in plans)
        )
        result = decode_document(document, expected="precise")
        self.assertTrue(result["success"])
        self.assertEqual(result["decoded_message"], "PRECISE")

    def test_full_requested_phrase_round_trip_and_rows(self) -> None:
        phrase = "Precise Intelligent Chemistry"
        document, plans, _ = make_document(phrase)
        self.assertEqual(
            [plan["plaintext"] for plan in plans],
            [
                "PREC",
                "ISE ",
                "INTE",
                "LLIG",
                "ENT ",
                "CHEM",
                "ISTR",
                "Y",
            ],
        )
        self.assertTrue(all(len(plan["cells"]) == 8 for plan in plans))
        result = decode_document(document, expected=phrase)
        self.assertTrue(result["matches_expected"])
        self.assertEqual(result["decoded_message"], phrase.upper())

    def test_spaces_and_fixed_count_row_framing_round_trip(self) -> None:
        for phrase in ("ABC D", "AB C", "ABCD X Y"):
            with self.subTest(phrase=phrase):
                document, _, _ = make_document(phrase)
                result = decode_document(document)
                self.assertEqual(result["decoded_message"], phrase)

    def test_final_partial_row_count_controls_decoding(self) -> None:
        document, plans, _ = make_document("ABCDE")
        self.assertEqual(plans[-1]["payload_character_count"], 1)
        self.assertEqual(plans[-1]["morse_character_count"], 1)
        self.assertEqual(plans[-1]["ascii_character_count"], 0)
        self.assertEqual(plans[-1]["colour_stream"], ["B", "R", "B", "B"])
        self.assertEqual(
            plans[-1]["colour_slots"],
            ["B", "R", "B", "B", "B", "B", "B", "B"],
        )
        self.assertEqual(plans[-1]["intensity_slots"], [1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(len(plans[-1]["cells"]), 8)
        result = decode_document(document)
        self.assertEqual(result["decoded_message"], "ABCDE")

    def test_malformed_row_count_is_rejected(self) -> None:
        document, _, _ = make_document("ABCD")
        document["rows"][0]["ascii_character_count"] = 1
        with self.assertRaisesRegex(ProtocolError, "channel counts"):
            decode_document(document)

    def test_every_row_requires_exactly_eight_configurations(self) -> None:
        document, _, _ = make_document("AB")
        document["rows"][0]["cells"].pop()
        document["rows"][0]["cell_count"] = 7
        with self.assertRaisesRegex(ProtocolError, "cell_count must be 8"):
            decode_document(document)

    def test_each_morse_character_uses_an_independent_four_slot_subblock(self) -> None:
        document, plans, _ = make_document("PRER")
        self.assertEqual(
            plans[0]["morse_colour_payloads"],
            [["B", "G", "G", "B"], ["B", "R"]],
        )
        self.assertEqual(
            plans[0]["colour_stream"],
            ["B", "G", "G", "B", "B", "R", "B", "B"],
        )
        self.assertEqual(plans[0]["colour_slots"], plans[0]["colour_stream"])
        result = decode_document(document, expected="PRER")
        self.assertTrue(result["matches_expected"])

    def test_measured_intensity_crossing_interval_is_rejected(self) -> None:
        document, _, _ = make_document("AB")
        measured = document["rows"][0]["cells"][0]
        intended_digit = measured["intended_pair"]["intensity_digit"]
        measured["observables"]["peak_cd_intensity"] = FAKE_INTENSITY[(intended_digit + 1) % 4]
        with self.assertRaisesRegex(ProtocolError, "not intended pair"):
            decode_document(document)

    def test_non_red_measured_cell_is_rejected(self) -> None:
        document, _, _ = make_document("AB")
        measured = document["rows"][0]["cells"][0]
        measured["material_parameters"]["is_red"] = 0
        with self.assertRaisesRegex(ProtocolError, "red phase"):
            decode_document(document)

    def test_exact_phase_pair_is_preferred_over_larger_partial_proxy(self) -> None:
        red = {
            "color": "B",
            "gray": 5.0,
            "thickness": 48.0,
            "stretch": 100.0,
            "peak_cd_intensity": -6000.0,
            "intensity_digit": 0,
        }
        exact = {
            **red,
            "is_red": False,
            "peak_cd_intensity": 1000.0,
            "intensity_digit": 2,
        }
        partial = {
            **exact,
            "gray": 1.0,
            "peak_cd_intensity": 12000.0,
            "intensity_digit": 3,
        }
        proxy, kind = best_phase_proxy(
            red,
            {("B", 5.0, 48.0, 100.0): [exact]},
            {("B", 48.0, 100.0): [partial]},
        )
        self.assertIs(proxy, exact)
        self.assertEqual(kind, "exact_same_color_gray_thickness_stretch")

    def test_markdown_reports_round_trip(self) -> None:
        document, plans, library = make_document("Precise")
        result = decode_document(document)
        markdown = render_configuration_sequence_markdown("PRECISE", library, plans, result)
        self.assertIn("Message `PRECISE`", markdown)
        self.assertIn("Round-trip verification PASS", markdown)


if __name__ == "__main__":
    unittest.main()
