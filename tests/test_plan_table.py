"""Tests for plan tables and the plan-row fields of Action."""

import pytest

from planit_py.vocabulary import (
    AbstractCostModel,
    AbstractPolicy,
    Action,
    FixedDuration,
    Mode,
)


class TestActionFields:
    """The plan-row fields of Action are additive and keyword-only."""

    def test_existing_constructions_unchanged(self):
        """Keyword and positional constructions without the new fields still work.

        Fails if a new field lost its default, changed a default meaning, or
        shifted the positional order of the original five fields.
        """
        by_keyword = Action(
            target_id=3,
            mode=Mode.IFS,
            bands=("700nm",),
            integration=FixedDuration(duration_d=2.0),
            start_time_d=5.0,
        )
        by_position = Action(3, Mode.IFS, ("700nm",), FixedDuration(2.0), 5.0)
        for action in (by_keyword, by_position):
            assert action.target_id == 3
            assert action.start_time_d == 5.0
            assert action.asset_id == 0
            assert action.channel_ids == ()
            assert action.group_id == -1
            assert action.transition_kind == 0

    def test_new_fields_are_keyword_only(self):
        """The new fields set by keyword and refuse positional arguments.

        Fails if a new field became positional, which would let a sixth
        positional argument silently land in it.
        """
        action = Action(
            3,
            Mode.IMAGING,
            ("500nm",),
            FixedDuration(1.0),
            0.0,
            asset_id=1,
            channel_ids=(0, 2),
            group_id=4,
            transition_kind=1,
        )
        assert (action.asset_id, action.channel_ids) == (1, (0, 2))
        assert (action.group_id, action.transition_kind) == (4, 1)
        with pytest.raises(TypeError):
            Action(3, Mode.IMAGING, ("500nm",), FixedDuration(1.0), 0.0, 1)


def test_protocol_versions():
    """The cost-model protocol is at 2 and the policy protocol stays at 1.

    Fails if a version moved without its contract changing, or a contract
    change shipped without its version: cost models consume the new action
    fields, while policies only produce actions whose defaults keep their
    meaning.
    """
    assert AbstractCostModel.PROTOCOL_VERSION == 2
    assert AbstractPolicy.PROTOCOL_VERSION == 1
