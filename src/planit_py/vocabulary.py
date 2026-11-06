"""The shared decision vocabulary for mission scheduling.

These are the record types and abstract interfaces that a scheduling agent and
its environment (a mission simulator, an external survey simulator adapter, or
real operations) must agree on: what an action is, what the agent can see about
observability, how actions are priced, and what a policy looks like. planit
owns this vocabulary; environments import it, never the reverse.

Time convention: mission time is measured in days, and every duration or epoch
field carries the ``_d`` suffix.
"""

import abc
import enum
from typing import Any, ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp


class Mode(enum.Enum):
    """Observing mode of an action."""

    IMAGING = "imaging"
    IFS = "ifs"


class AbstractIntegrationPolicy(eqx.Module):
    """How an observation decides its integration time.

    An action carries an integration policy rather than a fixed duration so
    that adaptive stopping rules are expressible. The environment executes the
    policy and reports the realized duration back; costs charged to the
    mission ledger always use the realized value.
    """


@final
class FixedDuration(AbstractIntegrationPolicy):
    """Integrate for a predetermined duration.

    Attributes:
        duration_d: Integration time in days.
    """

    duration_d: float


@final
class StopOnCertificate(AbstractIntegrationPolicy):
    """Integrate until a certificate statistic crosses a threshold.

    Attributes:
        threshold: Certificate statistic value that stops the integration.
        max_duration_d: Upper bound on the integration time in days.
    """

    threshold: float
    max_duration_d: float


@final
class StopOnInformationRate(AbstractIntegrationPolicy):
    """Integrate until the information rate drops below a floor.

    Attributes:
        floor_nats_per_d: Information rate (nats per day) below which the
            integration stops.
        max_duration_d: Upper bound on the integration time in days.
    """

    floor_nats_per_d: float
    max_duration_d: float


@final
class Action(eqx.Module):
    """One scheduled observation.

    Grows by keyword-only field additions with defaults, so existing call
    sites stay valid and an action that sets none of the newer fields means
    what it meant before they existed. The newer fields carry the identity a
    row of a fixed-shape plan carries, so an action and a plan row describe
    the same observation.

    Attributes:
        target_id: Identifier of the target in the mission catalog.
        mode: Observing mode.
        bands: Names of the spectral bands or channels observed.
        integration: The integration policy; the environment reports the
            realized duration.
        start_time_d: Requested start epoch in days of mission time.
        asset_id: Index of the observing asset (a telescope, instrument
            platform or other facility) that executes the action. Needed once
            a campaign has more than one asset; 0 is the environment's first or
            only asset, so single-asset environments ignore it.
        channel_ids: Identifiers of the measurement channels the action
            produces, as declared by the environment. Needed because one
            pointing can feed several channels at once, and an integer set
            is what fixed-shape plan tables store. Empty (the default) means
            the environment derives the channels from ``mode`` and ``bands``.
        group_id: Coordination group. Actions sharing a non-negative group id
            are executed together or not at all, which a coordinated
            observation across channels or assets needs. -1 (the default)
            means the action stands alone.
        transition_kind: Identifier of the transition model that moves the
            asset from its previous action to this one, for example an
            ordinary slew versus a retarget that moves a separate vehicle.
            Needed because the cost of a leg can depend on how it is flown,
            not only on where it ends. 0 (the default) is the asset's default
            transition.
    """

    target_id: int
    mode: Mode
    bands: tuple[str, ...]
    integration: AbstractIntegrationPolicy
    start_time_d: float
    asset_id: int = eqx.field(default=0, kw_only=True)
    channel_ids: tuple[int, ...] = eqx.field(default=(), kw_only=True)
    group_id: int = eqx.field(default=-1, kw_only=True)
    transition_kind: int = eqx.field(default=0, kw_only=True)


@final
class ObservingContext(eqx.Module):
    """What the agent may know about observability at one epoch.

    Assembled by the environment (geometry, keepout, slew costs) and consumed
    by policies and cost models. Grows by keyword-only field additions so
    existing call sites stay valid.

    Attributes:
        time_d: Epoch in days of mission time.
        target_ids: Integer array of catalog target identifiers, shape (n,).
        observable: Boolean array, shape (n,); True where the target is
            observable at this epoch.
        slew_d: Float array, shape (n,); slew-plus-settle time in days to
            reach each target from the current attitude.
    """

    time_d: float
    target_ids: jax.Array = eqx.field(converter=jnp.asarray)
    observable: jax.Array = eqx.field(converter=jnp.asarray)
    slew_d: jax.Array = eqx.field(converter=jnp.asarray)


class AbstractCostModel(eqx.Module):
    """Prices actions in named mission resources.

    One cost model instance is shared by the agent and the environment: the
    agent ranks candidates with :meth:`expected`, and the environment charges
    the mission ledger with :meth:`realized` once the realized duration is
    known. Sharing the instance makes it impossible for the agent to optimize
    a different cost than the ledger charges.

    Costs are dictionaries mapping resource names to amounts; ``"time_d"`` is
    the universal resource and is always present.

    Protocol versions:
        1: actions carry a target, mode, bands, integration policy and start
            epoch.
        2: actions also carry ``asset_id``, ``channel_ids``, ``group_id`` and
            ``transition_kind``, and a cost model prices them: the asset
            decides which resources are charged, the channels what each
            exposure charges, and the transition kind what the leg into the
            action costs. A version 1 model ignores those fields and would
            misprice an action that sets them, so an environment that emits
            them requires version 2.
    """

    PROTOCOL_VERSION: ClassVar[int] = 2

    @abc.abstractmethod
    def expected(self, action: Action, ctx: ObservingContext) -> dict[str, float]:
        """Return the expected cost of an action before it is executed."""

    @abc.abstractmethod
    def realized(
        self, action: Action, ctx: ObservingContext, duration_d: float
    ) -> dict[str, float]:
        """Return the cost to charge given the realized integration duration."""


class AbstractPolicy(eqx.Module):
    """A scheduling policy: belief in, next action out.

    Policies are stateless and pure: everything they know arrives through the
    belief, the observing context, and the cost model, so the same policy runs
    unchanged inside a simulator or against a real mission. ``belief`` is
    whatever belief container the environment maintains; policies constrain it
    only through the operations they perform on it.

    Protocol versions:
        1: ``propose(belief, ctx, cost_model)`` returns one action. The newer
            keyword-only action fields did not change this contract: a policy
            that never sets them returns actions with their defaults, whose
            meaning is unchanged.
    """

    PROTOCOL_VERSION: ClassVar[int] = 1

    @abc.abstractmethod
    def propose(
        self, belief: Any, ctx: ObservingContext, cost_model: AbstractCostModel
    ) -> Action:
        """Return the next action to execute."""
