"""Tests for the decision vocabulary types."""

import equinox as eqx
import jax.numpy as jnp
import pytest

from planit_py.vocabulary import (
    AbstractCostModel,
    AbstractPolicy,
    Action,
    FixedDuration,
    Mode,
    ObservingContext,
    StopOnCertificate,
    StopOnInformationRate,
)


def make_action(**overrides):
    """Build a valid Action, with keyword overrides for individual fields."""
    fields = {
        "target_id": 7,
        "mode": Mode.IMAGING,
        "bands": ("500nm",),
        "integration": FixedDuration(duration_d=1.5),
        "start_time_d": 10.0,
    }
    fields.update(overrides)
    return Action(**fields)


class TestRecords:
    """Construction and immutability of the record types."""

    def test_action_construction(self):
        """An Action holds its fields as given."""
        action = make_action()
        assert action.target_id == 7
        assert action.mode is Mode.IMAGING
        assert action.integration.duration_d == 1.5

    def test_integration_policy_variants(self):
        """All three integration policies construct with their parameters."""
        cert = StopOnCertificate(threshold=5.0, max_duration_d=14.0)
        rate = StopOnInformationRate(floor_nats_per_d=0.01, max_duration_d=14.0)
        assert cert.threshold == 5.0
        assert rate.max_duration_d == 14.0

    def test_action_is_frozen(self):
        """Direct field assignment is rejected after construction."""
        action = make_action()
        with pytest.raises(AttributeError):
            action.target_id = 8

    def test_action_functional_update(self):
        """tree_at produces an updated copy without touching the original."""
        action = make_action()
        moved = eqx.tree_at(lambda a: a.start_time_d, action, 20.0)
        assert moved.start_time_d == 20.0
        assert action.start_time_d == 10.0

    def test_observing_context_converts_arrays(self):
        """List inputs are converted to arrays at construction."""
        ctx = ObservingContext(
            time_d=0.0,
            target_ids=[1, 2, 3],
            observable=[True, False, True],
            slew_d=[0.1, 0.2, 0.3],
        )
        assert ctx.target_ids.shape == (3,)
        assert bool(jnp.all(ctx.slew_d > 0))


class TestAbstractInterfaces:
    """The policy and cost-model interfaces enforce their contracts."""

    def test_abstract_types_do_not_instantiate(self):
        """The ABCs cannot be constructed directly."""
        with pytest.raises(TypeError):
            AbstractPolicy()
        with pytest.raises(TypeError):
            AbstractCostModel()

    def test_concrete_policy_proposes(self):
        """A minimal concrete policy returns an Action from propose()."""

        class FirstTarget(AbstractPolicy):
            """Always proposes the first observable target."""

            duration_d: float = 1.0

            def propose(self, belief, ctx, cost_model):
                """Propose the first target id with a fixed duration."""
                target = int(ctx.target_ids[0])
                return Action(
                    target_id=target,
                    mode=Mode.IMAGING,
                    bands=("500nm",),
                    integration=FixedDuration(duration_d=self.duration_d),
                    start_time_d=ctx.time_d,
                )

        ctx = ObservingContext(
            time_d=3.0,
            target_ids=[4, 5],
            observable=[True, True],
            slew_d=[0.0, 0.0],
        )
        action = FirstTarget().propose(belief=None, ctx=ctx, cost_model=None)
        assert action.target_id == 4
        assert action.start_time_d == 3.0
