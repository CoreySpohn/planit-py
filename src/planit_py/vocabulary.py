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
    row of a :class:`PlanTable` carries, so an action and a plan row describe
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


def _as_days(x):
    """Convert to an array of days, promoting integer input to floating point."""
    a = jnp.asarray(x)
    if jnp.issubdtype(a.dtype, jnp.integer):
        a = a.astype(float)
    return a


@final
class PlanTable(eqx.Module):
    """A plan as data: a fixed-capacity table of scheduled observations.

    Each valid row is one observation, with the same identity an
    :class:`Action` carries: which target, which asset executes it, which
    measurement channels it produces, which coordination group it belongs
    to, when it starts, and how the asset reaches it. A planner writes a
    plan table once; an executor reads it row by row, converting start
    epochs to its own clock, without running the planner again.

    Every array has the fixed leading length ``capacity`` so that tables of
    different plans with the same capacity share one shape. Valid rows come
    first, in nondecreasing start order; the remaining padding rows hold the
    fill values (ids 0, group -1, start 0.0, no channels), so two tables that
    hold the same plan at the same capacity are equal leaf by leaf.

    Start epochs must be float64, which needs JAX's 64-bit mode
    (``jax_enable_x64``). An executor that converts days to a fine clock
    needs that resolution: the float32 spacing at day 270 is about 2.6 s.
    Build tables on the host; ``__check_init__`` reads concrete values.

    Attributes:
        target_ids: Integer array, shape (capacity,); catalog identifier of
            the target observed by each row.
        asset_ids: Integer array, shape (capacity,); index of the asset (a
            telescope, instrument platform or other facility) that executes
            each row.
        channel_ids: Integer array, shape (n_channels,); the distinct channel
            identifier of each column of ``channel_mask``.
        channel_mask: Boolean array, shape (capacity, n_channels); True where
            the row produces that column's channel. A valid row produces at
            least one channel.
        group_ids: Integer array, shape (capacity,); coordination group of
            each row. Rows sharing a non-negative group id are executed
            together or not at all; -1 means the row stands alone.
        start_times_d: Float64 array, shape (capacity,); requested start epoch
            of each row in days of mission time.
        transition_kinds: Integer array, shape (capacity,); identifier of the
            transition model that moves the row's asset from its previous
            observation to this one; 0 is the asset's default transition.
        valid: Boolean array, shape (capacity,); True for the rows that hold
            observations. Valid rows form a prefix of the table.
    """

    target_ids: jax.Array = eqx.field(converter=jnp.asarray)
    asset_ids: jax.Array = eqx.field(converter=jnp.asarray)
    channel_ids: jax.Array = eqx.field(converter=jnp.asarray)
    channel_mask: jax.Array = eqx.field(converter=jnp.asarray)
    group_ids: jax.Array = eqx.field(converter=jnp.asarray)
    start_times_d: jax.Array = eqx.field(converter=_as_days)
    transition_kinds: jax.Array = eqx.field(converter=jnp.asarray)
    valid: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Refuse tables whose shapes, dtypes or row values break the contract."""
        row_fields = {
            "target_ids": self.target_ids,
            "asset_ids": self.asset_ids,
            "group_ids": self.group_ids,
            "start_times_d": self.start_times_d,
            "transition_kinds": self.transition_kinds,
            "valid": self.valid,
        }
        capacity = self.valid.shape[0] if self.valid.ndim == 1 else 0
        if capacity < 1:
            raise ValueError(
                f"valid must be a 1-D array of at least one row, got shape "
                f"{self.valid.shape}"
            )
        for name, a in row_fields.items():
            if a.shape != (capacity,):
                raise ValueError(
                    f"{name} has shape {a.shape}; every row field needs shape "
                    f"({capacity},)"
                )
        if self.channel_ids.ndim != 1 or self.channel_ids.shape[0] < 1:
            raise ValueError(
                f"channel_ids must be a 1-D array of at least one channel id, "
                f"got shape {self.channel_ids.shape}"
            )
        n_channels = self.channel_ids.shape[0]
        if self.channel_mask.shape != (capacity, n_channels):
            raise ValueError(
                f"channel_mask has shape {self.channel_mask.shape}; expected "
                f"({capacity}, {n_channels})"
            )

        for name in ("target_ids", "asset_ids", "group_ids", "transition_kinds"):
            if not jnp.issubdtype(row_fields[name].dtype, jnp.integer):
                raise TypeError(f"{name} must be integer, got {row_fields[name].dtype}")
        if not jnp.issubdtype(self.channel_ids.dtype, jnp.integer):
            raise TypeError(
                f"channel_ids must be integer, got {self.channel_ids.dtype}"
            )
        for name, a in (("channel_mask", self.channel_mask), ("valid", self.valid)):
            if a.dtype != jnp.bool_:
                raise TypeError(f"{name} must be boolean, got {a.dtype}")
        if self.start_times_d.dtype != jnp.float64:
            raise TypeError(
                f"start_times_d must be float64, got {self.start_times_d.dtype}; "
                "enable 64-bit mode with jax.config.update('jax_enable_x64', True)"
            )

        if bool(jnp.any(self.channel_ids < 0)):
            raise ValueError("channel_ids must be non-negative")
        ordered = jnp.sort(self.channel_ids)
        if bool(jnp.any(ordered[1:] == ordered[:-1])):
            raise ValueError("channel_ids must be distinct")

        n_rows = jnp.sum(self.valid)
        if bool(jnp.any(self.valid != (jnp.arange(capacity) < n_rows))):
            raise ValueError("valid rows must come first, before every padding row")
        live = self.valid
        for name in ("target_ids", "asset_ids", "transition_kinds"):
            if bool(jnp.any(live & (row_fields[name] < 0))):
                raise ValueError(f"{name} must be non-negative on valid rows")
        if bool(jnp.any(live & (self.group_ids < -1))):
            raise ValueError("group_ids must be -1 or non-negative on valid rows")
        start = self.start_times_d
        if bool(jnp.any(live & ~jnp.isfinite(start))):
            raise ValueError("start_times_d must be finite on valid rows")
        if bool(jnp.any(live[1:] & (start[1:] < start[:-1]))):
            raise ValueError("start_times_d must be nondecreasing over the valid rows")
        if bool(jnp.any(live & ~jnp.any(self.channel_mask, axis=1))):
            raise ValueError("every valid row must produce at least one channel")
        filled = (
            (self.target_ids != 0)
            | (self.asset_ids != 0)
            | (self.group_ids != -1)
            | (start != 0.0)
            | (self.transition_kinds != 0)
            | jnp.any(self.channel_mask, axis=1)
        )
        if bool(jnp.any(~live & filled)):
            raise ValueError(
                "padding rows must hold the fill values (ids 0, group -1, "
                "start 0.0, no channels)"
            )

    @property
    def n_rows(self) -> jax.Array:
        """Number of valid rows, as an integer scalar array."""
        return jnp.sum(self.valid)

    @classmethod
    def from_lists(
        cls,
        target_ids,
        start_times_d,
        channel_sets,
        *,
        channel_ids=None,
        asset_ids=None,
        group_ids=None,
        transition_kinds=None,
        capacity=None,
    ) -> "PlanTable":
        """Build a plan table from per-row Python lists in chronological order.

        Args:
            target_ids: Catalog target identifier of each row.
            start_times_d: Start epoch of each row in days of mission time.
            channel_sets: For each row, an iterable of the channel ids it
                produces.
            channel_ids: Column order of ``channel_mask``. Defaults to the
                sorted distinct channel ids named in ``channel_sets``.
            asset_ids: Asset index of each row; defaults to 0 for every row.
            group_ids: Coordination group of each row; defaults to -1 (each
                row stands alone).
            transition_kinds: Transition model of each row; defaults to 0.
            capacity: Number of rows in the table, at least the number of
                observations; defaults to that number. Extra rows are padding.

        Returns:
            The plan table, validated.

        Raises:
            ValueError: If the lists differ in length, a row names a channel
                that ``channel_ids`` does not declare, ``capacity`` is smaller
                than the number of rows, or the table fails validation.
        """
        n = len(target_ids)
        channel_sets = [tuple(row) for row in channel_sets]
        columns = {
            "start_times_d": list(start_times_d),
            "channel_sets": channel_sets,
            "asset_ids": [0] * n if asset_ids is None else list(asset_ids),
            "group_ids": [-1] * n if group_ids is None else list(group_ids),
            "transition_kinds": (
                [0] * n if transition_kinds is None else list(transition_kinds)
            ),
        }
        for name, column in columns.items():
            if len(column) != n:
                raise ValueError(
                    f"{name} has {len(column)} entries; target_ids has {n}"
                )
        if channel_ids is None:
            channel_ids = sorted({c for row in channel_sets for c in row})
        channel_ids = list(channel_ids)
        capacity = n if capacity is None else capacity
        if capacity < n:
            raise ValueError(f"capacity {capacity} is smaller than the {n} rows")

        column_of = {c: j for j, c in enumerate(channel_ids)}
        mask = []
        for i, row in enumerate(channel_sets):
            flags = [False] * len(channel_ids)
            for c in row:
                if c not in column_of:
                    raise ValueError(
                        f"row {i} names channel {c}, which channel_ids does not declare"
                    )
                flags[column_of[c]] = True
            mask.append(flags)

        pad = capacity - n
        return cls(
            target_ids=list(target_ids) + [0] * pad,
            asset_ids=columns["asset_ids"] + [0] * pad,
            channel_ids=channel_ids,
            channel_mask=mask + [[False] * len(channel_ids)] * pad,
            group_ids=columns["group_ids"] + [-1] * pad,
            start_times_d=columns["start_times_d"] + [0.0] * pad,
            transition_kinds=columns["transition_kinds"] + [0] * pad,
            valid=[True] * n + [False] * pad,
        )

    @classmethod
    def from_actions(cls, actions, *, channel_ids=None, capacity=None) -> "PlanTable":
        """Build a plan table from actions in chronological order.

        Each row keeps its action's target, start epoch, asset, channels,
        group and transition kind. Mode, bands and the integration policy are
        not part of a plan row: an executor resolves them from the channels.

        Args:
            actions: The actions, in nondecreasing start order.
            channel_ids: Column order of ``channel_mask``; see
                :meth:`from_lists`.
            capacity: Number of rows in the table; see :meth:`from_lists`.

        Returns:
            The plan table, validated.

        Raises:
            ValueError: If an action declares no ``channel_ids``, or as
                :meth:`from_lists` raises.
        """
        actions = list(actions)
        for i, action in enumerate(actions):
            if not action.channel_ids:
                raise ValueError(
                    f"action {i} declares no channel_ids; a plan row needs its channels"
                )
        return cls.from_lists(
            [a.target_id for a in actions],
            [a.start_time_d for a in actions],
            [a.channel_ids for a in actions],
            channel_ids=channel_ids,
            asset_ids=[a.asset_id for a in actions],
            group_ids=[a.group_id for a in actions],
            transition_kinds=[a.transition_kind for a in actions],
            capacity=capacity,
        )


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
