import pandas as pd
from types import SimpleNamespace

import ms3.operations as operations
from ms3.operations import (
    make_coloring_reports_and_warnings,
    missing_replacement_tones,
    replacement_tone_evidence,
    validate_replacement_tones,
)


def replacement_row(**overrides):
    values = {
        "numeral": "bVI",
        "form": pd.NA,
        "figbass": pd.NA,
        "changes": "^#2",
        "relativeroot": "III",
        "localkey_is_minor": True,
        "root": -5,
        "chord_tones": (-5, -4, 11),
        "observed_tpcs": (-5, -4, 4),
    }
    return pd.Series(values | overrides)


def test_absent_replacement_tone_is_independent_of_global_mismatch_ratio():
    assert missing_replacement_tones(replacement_row()) == (11,)


def test_sounding_replacement_tone_does_not_warn():
    assert missing_replacement_tones(
        replacement_row(observed_tpcs=(-5, -4, 11))
    ) == ()


def test_row_without_tone_replacement_does_not_warn():
    assert missing_replacement_tones(replacement_row(changes=pd.NA)) == ()


def test_unexpandable_label_returns_structured_unavailable_evidence():
    evidence = replacement_tone_evidence(
        replacement_row(numeral=pd.NA, root=pd.NA)
    )

    assert evidence["replacement_validation_status"] == "no_replacement_tones_asserted"
    assert evidence["expansion_reproduction_status"] == "stored_expansion_unavailable"


def test_unavailable_score_segment_does_not_become_an_absence_warning():
    row = replacement_row(observed_tpcs=pd.NA)

    assert replacement_tone_evidence(row)["replacement_validation_status"] == (
        "score_segment_unavailable"
    )
    assert missing_replacement_tones(row) == ()


def test_evidence_preserves_both_asserted_and_observed_interpretations():
    evidence = replacement_tone_evidence(replacement_row())

    assert evidence == {
        "replacement_validation_status": "replacement_tone_absent_from_score_segment",
        "expansion_reproduction_status": "stored_expansion_matches_current_ms3",
        "stored_chord_tpcs": (-5, -4, 11),
        "recomputed_chord_tpcs": (-5, -4, 11),
        "asserted_replacement_tpcs": (11,),
        "observed_tpcs": (-5, -4, 4),
        "missing_replacement_tpcs": (11,),
        "replacement_score_evidence_scope": "unspecified",
    }


def test_added_tones_are_not_misclassified_as_replacements():
    assert missing_replacement_tones(
        replacement_row(changes="+6", observed_tpcs=(-5, -4, 4))
    ) == ()


def test_changed_stored_expansion_is_reported_without_rewriting_it():
    evidence = replacement_tone_evidence(
        replacement_row(chord_tones=(-5, -4, 4))
    )

    assert (
        evidence["expansion_reproduction_status"]
        == "stored_expansion_differs_from_current_ms3"
    )
    assert evidence["stored_chord_tpcs"] == (-5, -4, 4)
    assert evidence["recomputed_chord_tpcs"] == (-5, -4, 11)


def test_table_validator_counts_sustained_and_delayed_replacement_tones():
    harmonies = pd.DataFrame(
        [
            replacement_row(
                quarterbeats=1,
                duration_qb=1,
                observed_tpcs=pd.NA,
            ),
            replacement_row(
                quarterbeats=3,
                duration_qb=1,
                observed_tpcs=pd.NA,
            ),
        ]
    )
    notes = pd.DataFrame(
        [
            {"quarterbeats": "", "duration_qb": "", "tpc": ""},
            {"quarterbeats": 0, "duration_qb": 2, "tpc": 11},
            {"quarterbeats": 3.5, "duration_qb": 0.5, "tpc": 11},
        ]
    )

    result = validate_replacement_tones(harmonies, notes)

    assert result.replacement_validation_status.tolist() == [
        "replacement_tones_confirmed_in_score_segment",
        "replacement_tones_confirmed_in_score_segment",
    ]
    assert result.observed_tpcs.tolist() == [(11,), (11,)]
    assert result.replacement_score_evidence_scope.tolist() == [
        "all_notes_overlapping_harmony_segment",
        "all_notes_overlapping_harmony_segment",
    ]


def test_table_validator_retains_absent_tone_as_machine_readable_warning():
    harmonies = pd.DataFrame(
        [replacement_row(quarterbeats=1, duration_qb=1, observed_tpcs=pd.NA)]
    )
    notes = pd.DataFrame(
        [{"quarterbeats": 1, "duration_qb": 1, "tpc": 4}]
    )

    result = validate_replacement_tones(harmonies, notes)

    assert result.loc[0, "asserted_replacement_tpcs"] == (11,)
    assert (
        result.loc[0, "expansion_reproduction_status"]
        == "stored_expansion_matches_current_ms3"
    )
    assert result.loc[0, "observed_tpcs"] == (4,)
    assert result.loc[0, "missing_replacement_tpcs"] == (11,)
    assert (
        result.loc[0, "replacement_validation_status"]
        == "replacement_tone_absent_from_score_segment"
    )


def test_review_report_writes_provenance_columns_and_fails_on_absent_tone(
    tmp_path, monkeypatch
):
    report = pd.DataFrame(
        [
            replacement_row(
                mc=12,
                mn=12,
                mc_onset=0,
                mn_onset=0,
                label="bVI(^#2)/III",
                count_ratio=0.0,
                added_tones=(),
                replacement_score_evidence_scope=(
                    "note_onsets_within_harmony_segment"
                ),
            )
        ]
    )

    class FakeParse:
        logger = operations.get_logger("replacement-validation-test")

        def color_non_chord_tones(self):
            return {("corpus", "piece"): [(SimpleNamespace(piece="piece"), report)]}

        def __getitem__(self, key):
            return SimpleNamespace(
                logger_names={"piece": "replacement-validation-test"}
            )

    written = []
    monkeypatch.setattr(operations, "get_ignored_warning_ids", lambda logger: set())
    monkeypatch.setattr(
        operations,
        "compute_path_from_file",
        lambda *args, **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        operations, "write_tsv", lambda dataframe, path: written.append(dataframe.copy())
    )

    assert make_coloring_reports_and_warnings(FakeParse()) is False
    assert written[0].loc[0, "missing_replacement_tpcs"] == (11,)
    assert written[0].loc[0, "replacement_validation_status"] == (
        "replacement_tone_absent_from_score_segment"
    )
    assert written[0].loc[0, "expansion_reproduction_status"] == (
        "stored_expansion_matches_current_ms3"
    )
    assert written[0].loc[0, "replacement_score_evidence_scope"] == (
        "note_onsets_within_harmony_segment"
    )
