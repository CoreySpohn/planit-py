"""Tests for plan tables and the plan-row fields of Action."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from planit_py.vocabulary import (
    AbstractCostModel,
    AbstractPolicy,
    Action,
    FixedDuration,
    Mode,
    PlanTable,
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

    Fails if either constant drifts from the release that added the plan-row
    action fields: cost models must honor those fields when pricing, while
    the calling contract of a policy did not change.
    """
    assert AbstractCostModel.PROTOCOL_VERSION == 2
    assert AbstractPolicy.PROTOCOL_VERSION == 1


IMAGING, IFS = 0, 1
A, B, C = 0, 1, 2
SLOTS_D = [0.0, 90.0, 180.0, 270.0]
PLANS = {
    "myopic": ([A, A, B, C], [[IMAGING], [IFS], [IMAGING], [IMAGING]]),
    "lookahead": ([A, C, A, C], [[IMAGING], [IMAGING], [IFS], [IFS]]),
}


def table_fields(**overrides):
    """Fields of a valid table: two observations and one padding row."""
    fields = {
        "target_ids": [0, 2, 0],
        "asset_ids": [0, 1, 0],
        "channel_ids": [IMAGING, IFS],
        "channel_mask": [[True, False], [True, True], [False, False]],
        "group_ids": [-1, 3, -1],
        "start_times_d": [0.0, 90.0, 0.0],
        "transition_kinds": [0, 1, 0],
        "valid": [True, True, False],
    }
    fields.update(overrides)
    return fields


def channel_sets(table):
    """The channel ids of each valid row, read back from the mask."""
    n = int(table.n_rows)
    ids = table.channel_ids
    return [tuple(int(c) for c in ids[table.channel_mask[i]]) for i in range(n)]


class TestPlanTableConstruction:
    """Construction helpers agree and validation refuses broken tables."""

    def test_from_lists_matches_direct_construction(self):
        """from_lists builds the same leaves as the explicit fields.

        Fails if from_lists maps a channel to the wrong column, drops a
        per-row field, or pads with values other than the fill values.
        """
        direct = PlanTable(**table_fields())
        built = PlanTable.from_lists(
            [0, 2],
            [0.0, 90],
            [[IMAGING], [IFS, IMAGING]],
            channel_ids=[IMAGING, IFS],
            asset_ids=[0, 1],
            group_ids=[-1, 3],
            transition_kinds=[0, 1],
            capacity=3,
        )
        assert eqx.tree_equal(direct, built)
        assert built.start_times_d.dtype == jnp.float64
        assert int(built.n_rows) == 2
        empty = PlanTable.from_lists([], [], [], channel_ids=[IMAGING], capacity=2)
        assert int(empty.n_rows) == 0

    def test_refuses_float32_days_when_64_bit_mode_is_off(self):
        """With 64-bit mode off, construction fails and names the fix.

        Fails if a table silently stores float32 start epochs, which cannot
        resolve seconds late in a mission, when the caller passed float64.
        """
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(TypeError, match="64-bit mode is off"):
                PlanTable.from_lists([0], [270.0], [[IMAGING]], channel_ids=[IMAGING])
        finally:
            jax.config.update("jax_enable_x64", True)

    @pytest.mark.parametrize(
        ("overrides", "error"),
        [
            ({"target_ids": [0, 2]}, ValueError),
            ({"channel_mask": [[True], [True], [False]]}, ValueError),
            ({"target_ids": [0.0, 2.0, 0.0]}, TypeError),
            ({"valid": [1, 1, 0]}, TypeError),
            (
                {"start_times_d": jnp.asarray([0.0, 90.0, 0.0], jnp.float32)},
                TypeError,
            ),
        ],
        ids=["row-length", "mask-width", "float-ids", "int-valid", "float32-days"],
    )
    def test_refuses_bad_structure(self, overrides, error):
        """Shapes and dtypes outside the contract are refused at construction.

        Fails if a mismatched shape or a lossy dtype (float ids, float32 start
        epochs that cannot resolve seconds late in a mission) is accepted.
        """
        with pytest.raises(error):
            PlanTable(**table_fields(**overrides))

    @pytest.mark.parametrize(
        ("overrides", "rule"),
        [
            ({"valid": [False, True, True]}, "come first"),
            ({"start_times_d": [90.0, 0.0, 0.0]}, "nondecreasing"),
            (
                {"channel_mask": [[True, False], [False, False], [False, False]]},
                "at least one channel",
            ),
            ({"target_ids": [0, 2, 5]}, "fill values"),
            ({"channel_ids": [IFS, IFS]}, "distinct"),
        ],
        ids=[
            "valid-not-prefix",
            "start-decreasing",
            "row-without-channel",
            "padding-not-filled",
            "duplicate-channel",
        ],
    )
    def test_refuses_broken_plan_structure(self, overrides, rule):
        """Rows that break the plan's ordering or set semantics are refused.

        Fails if a table can interleave padding with observations, run
        backwards in time, schedule a row that produces nothing, hide data in
        padding (so equal plans compare unequal), or repeat a channel column.
        """
        with pytest.raises(ValueError, match=rule):
            PlanTable(**table_fields(**overrides))

    @pytest.mark.parametrize(
        ("overrides", "rule"),
        [
            ({"asset_ids": [-1, 1, 0]}, "asset_ids must be non-negative"),
            ({"group_ids": [-2, 3, -1]}, "group_ids must be -1 or"),
            ({"start_times_d": [-1.0, 90.0, 0.0]}, "start_times_d must be non-neg"),
            ({"start_times_d": [0.0, jnp.inf, 0.0]}, "finite"),
            ({"channel_ids": [-1, IFS]}, "channel_ids must be non-negative"),
        ],
        ids=[
            "negative-asset",
            "group-below-minus-one",
            "negative-start",
            "infinite-start",
            "negative-channel",
        ],
    )
    def test_refuses_out_of_range_values(self, overrides, rule):
        """Identifiers and start epochs outside their ranges are refused.

        Fails if a valid row can name a negative asset or channel, an
        undefined group, or a start before mission time zero or at infinity,
        any of which an executor would turn into a bad index or clock tick.
        """
        with pytest.raises(ValueError, match=rule):
            PlanTable(**table_fields(**overrides))

    @pytest.mark.parametrize(
        ("args", "kwargs"),
        [
            (([0, 1], [0.0], [[IMAGING], [IMAGING]]), {}),
            (([0], [0.0], [[IFS]]), {"channel_ids": [IMAGING]}),
            (([0, 1], [0.0, 1.0], [[IMAGING], [IMAGING]]), {"capacity": 1}),
            (([], [], []), {"channel_ids": [], "capacity": 2}),
        ],
        ids=[
            "length-mismatch",
            "undeclared-channel",
            "capacity-too-small",
            "no-channels",
        ],
    )
    def test_from_lists_refusals(self, args, kwargs):
        """from_lists refuses inputs it cannot place in a table.

        Fails if ragged lists are truncated, an undeclared channel is dropped,
        rows beyond the capacity are silently cut, or a campaign declares no
        channels.
        """
        kwargs = {"channel_ids": [IMAGING, IFS], **kwargs}
        with pytest.raises(ValueError):
            PlanTable.from_lists(*args, **kwargs)

    def test_from_actions_keeps_plan_row_fields(self):
        """Actions convert to rows carrying their plan-row fields.

        Fails if from_actions drops or misroutes the asset, channels, group or
        transition kind, or accepts an action without channels.
        """

        def action(target, start_d, **kw):
            return Action(
                target, Mode.IMAGING, ("500nm",), FixedDuration(0.01), start_d, **kw
            )

        table = PlanTable.from_actions(
            [
                action(0, 0.0, channel_ids=(IMAGING,)),
                action(
                    2,
                    90.0,
                    asset_id=1,
                    channel_ids=(IFS, IMAGING),
                    group_id=3,
                    transition_kind=1,
                ),
            ],
            channel_ids=[IMAGING, IFS],
            capacity=3,
        )
        assert eqx.tree_equal(table, PlanTable(**table_fields()))
        with pytest.raises(ValueError, match="no channel_ids"):
            PlanTable.from_actions([action(0, 0.0)], channel_ids=[IMAGING])


@pytest.fixture(scope="module")
def tables():
    """Both plans of the four-slot campaign, at a capacity of six rows."""
    return {
        name: PlanTable.from_lists(
            targets, SLOTS_D, chans, channel_ids=[IMAGING, IFS], capacity=6
        )
        for name, (targets, chans) in PLANS.items()
    }


class TestPlanSequences:
    """A four-slot, three-target campaign planned two ways.

    Targets A, B and C are observed at days 0, 90, 180 and 270. A first visit
    to a target uses the imaging channel and a second the spectroscopy (IFS)
    channel. A one-step lookahead plans A, A, B, C; a deeper lookahead plans
    A, C, A, C.
    """

    def test_sequences_read_back(self, tables):
        """Each plan reads back its targets, start days and channel sets.

        Fails if a row is reordered, a start day changes, or a channel set
        comes back different, which would make an executor run another plan.
        """
        for name, (targets, chans) in PLANS.items():
            table = tables[name]
            n = int(table.n_rows)
            assert n == 4
            assert [int(t) for t in table.target_ids[:n]] == targets
            assert [float(t) for t in table.start_times_d[:n]] == SLOTS_D
            assert channel_sets(table) == [tuple(c) for c in chans]
            assert not bool(jnp.any(table.valid[n:]))
        assert not eqx.tree_equal(tables["myopic"], tables["lookahead"])

    def test_plans_share_one_compiled_program(self, tables):
        """Plans of one capacity and channel declaration share one program.

        Fails if plans with the same capacity and declared channels differ in
        structure or shape, including a plan that uses fewer of the channels,
        which would force an executor to recompile for every plan.
        """
        traces = []

        @eqx.filter_jit
        def spectroscopy_visits(table):
            traces.append(None)
            return jnp.sum(table.valid & table.channel_mask[:, IFS])

        assert int(spectroscopy_visits(tables["myopic"])) == 1
        assert int(spectroscopy_visits(tables["lookahead"])) == 2
        imaging_only = PlanTable.from_lists(
            [A, B],
            SLOTS_D[:2],
            [[IMAGING], [IMAGING]],
            channel_ids=[IMAGING, IFS],
            capacity=6,
        )
        assert int(spectroscopy_visits(imaging_only)) == 0
        assert len(traces) == 1
