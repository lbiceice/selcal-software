# SelCal v2 null, randomness, full-reselection, and calibration design

Status: `APPROVED_FOR_M0_M2_TDD_IMPLEMENTATION`

Date: 2026-08-27

Written approval: the author explicitly approved this v2 specification on
2026-08-27 with the instruction (translated from Chinese), “Approve the v2 specification; start the M0–M2 TDD implementation.” This
approval releases only the implementation gate defined in Section 14. It does
not constitute M0–M2 verification, M6 scientific validation, release approval,
or SoftwareX submission readiness.

Implementation clarification after the first adversarial plan review on
2026-08-27: the data-flow order in Section 8 remains normative. Null binding
occurs before the observed statistic scan so a disabled null can terminate
without a statistic scan. If the observed statistic scan fails analytically,
“zero null calls” means zero random-stream, token-sampling, token-application,
or replicate calls after the already completed bind; it does not mean zero
calls to `bind`. This clarification changes no null distribution, p-value, or
result field.

Scope: the next M0--M2 implementation slice after contract resolution and the
Pearson adapter. This document changes no runtime behavior by itself.

## 1. Decision

SelCal will not attach executable calibration semantics to the current
`selcal.scientific-plan.v1` identity. The current v1 calculator and its golden
digests remain available as historical, non-executable identities. Executable
null generation and post-selection calibration will use an explicit
`selcal.scientific-plan.v2` contract.

This is required because four choices affect the sampled distribution or the
reported p-value and were not uniquely fixed by v1:

1. whether the family decision statistic is the true maximum score or the
   score of a tolerance-selected representative;
2. the precise circular-shift state space;
3. the treatment of a short final block and the identity transformation;
4. the exact mapping from plan identity and replicate ID to a transformation.

No v1 digest may be silently reinterpreted. Migration from v1 to v2 is an
explicit user action that produces a new plan digest.

## 2. Evidence boundary

At the time of this design:

- immutable inputs, canonical JSON, v1 plan resolution, explicit adapter
  registries, common support, deterministic selection, binned NetTE, and
  lagged Pearson are implemented and locally tested;
- null transformations, replicate execution, calibrated p-values, evidence
  bundles, M6 scientific validation, UI, release, and the SoftwareX manuscript
  are not implemented or validated;
- passing this design and its future unit tests will prove algorithmic contract
  execution only. It will not prove that a null is scientifically valid for a
  particular data-generating process.

## 3. Scientific-plan v2 identity

V2 uses separate exact types: `PlanRequestV2`, `ResolvedScientificPlanV2`,
`ResolvedAdaptersV2`, and `PlanResolutionV2`. They do not subclass the v1
types. Only the v2 resolver can apply the private v2 seal. The v2 calibrator
checks `type(resolution) is PlanResolutionV2` and otherwise raises the typed
`PlanVersionError`; it never infers a plan version from an adapter name.

The canonical v2 payload has exactly the following top-level and nested fields.
Missing or extra fields fail closed:

```json
{
  "schema": "selcal.scientific-plan.v2",
  "statistic": {"name": "..._vN", "params": {}},
  "candidates": [1, 2, 3],
  "selection": {
    "rule": "max_upper|max_absolute",
    "tie_tolerance": 1e-12,
    "decision_contract": "family_max_with_canonical_tie_label_v2"
  },
  "null": {"name": "..._vN", "params": {}},
  "common_support": "max_candidate_lag_v1",
  "replicates": 999,
  "alpha": 0.05,
  "root_seed": 0,
  "rng_contract": "sha256_framed_plan_rid_stream_to_pcg64_raw64_v1",
  "calibration_contract": "full_reselection_global_mc_plus_one_v2",
  "failure_contract": "exact_b_fail_closed_v2"
}
```

The exact canonical spelling is normative. Object insertion order is not:
`canonical_json_bytes` recursively rejects unsupported values, encodes finite
floats by their hexadecimal float64 representation, sorts keys, and emits
UTF-8 with compact separators.

`root_seed` must be an exact built-in `int` in `[0, 2**64 - 1]`. Boolean,
NumPy integer, negative, and overflow values are rejected. The root seed is
committed through the v2 plan digest and is not encoded a second time in a
replicate seed preimage.

`replicates` must be an exact built-in `int` in `[1, 1_000_000]`. This is the
v2 plan representability cap used by schema validation, canonical hashing, and
migration; it is not the current in-memory executor cap. A plan may therefore
resolve and hash successfully but still be refused by the versioned execution
budget in Section 10A. A future executor may introduce another explicitly
versioned execution envelope; it may not change the v2 plan bytes or hashes
silently. `alpha` must be an exact built-in `float`, finite and strictly
between zero and one.

The v1 calculator, v1 payload bytes, and existing v1 golden digests remain
unchanged. A v1 object is not accepted by the v2 calibrator.

The first normative v2 known-answer plan is:

```json
{
  "schema": "selcal.scientific-plan.v2",
  "candidates": [1, 2, 3],
  "statistic": {"name": "lagged_pearson_v1", "params": {}},
  "selection": {
    "rule": "max_upper",
    "tie_tolerance": 1e-12,
    "decision_contract": "family_max_with_canonical_tie_label_v2"
  },
  "null": {"name": "circular_shift_v2", "params": {"min_shift": 1}},
  "replicates": 9,
  "alpha": 0.05,
  "root_seed": 17,
  "failure_contract": "exact_b_fail_closed_v2",
  "rng_contract": "sha256_framed_plan_rid_stream_to_pcg64_raw64_v1",
  "common_support": "max_candidate_lag_v1",
  "calibration_contract": "full_reselection_global_mc_plus_one_v2"
}
```

Its exact canonical UTF-8 bytes are:

```text
{"alpha":{"$float64":"0x1.999999999999ap-5"},"calibration_contract":"full_reselection_global_mc_plus_one_v2","candidates":[1,2,3],"common_support":"max_candidate_lag_v1","failure_contract":"exact_b_fail_closed_v2","null":{"name":"circular_shift_v2","params":{"min_shift":1}},"replicates":9,"rng_contract":"sha256_framed_plan_rid_stream_to_pcg64_raw64_v1","root_seed":17,"schema":"selcal.scientific-plan.v2","selection":{"decision_contract":"family_max_with_canonical_tie_label_v2","rule":"max_upper","tie_tolerance":{"$float64":"0x1.19799812dea11p-40"}},"statistic":{"name":"lagged_pearson_v1","params":{}}}
```

The SHA-256 digest is:

```text
dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931
```

## 4. Selection and family decision statistic

For candidate `c`, let the adapter return the signed estimate `e_c(D)`. The
central selector defines

```text
q_c(D) = e_c(D)       for max_upper
q_c(D) = abs(e_c(D))  for max_absolute
m(D)   = max_c q_c(D)
H(D)   = {c : m(D) - q_c(D) <= tie_tolerance}
s(D)   = the smallest canonical candidate in H(D)
A(D)   = m(D)
```

Consequences:

- `s(D)` is a deterministic label for reporting the selected candidate;
- `e_s(D)` remains signed and is reported separately;
- `A(D)` is always the true family maximum and is the only statistic used in
  the global tail comparison;
- tie tolerance never lowers `A(D)` and is never applied to the exceedance
  comparison;
- the tail is exactly `A_b >= A_observed`.

The existing v1 behavior, where a representative within tolerance can carry a
score slightly below the maximum, is retained only for v1 characterization and
migration tests. It is not executable calibration semantics.

## 5. Statistic adapter boundary

The bound statistic contract owns the frozen candidate tuple. It exposes one
full-vector operation and does not accept a caller-supplied subset or selection
rule:

```python
class BoundStatisticAdapter(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    @property
    def candidates(self) -> tuple[int, ...]: ...

    @property
    def backend_identity(self) -> str: ...

    @property
    def preprocessing_identity(self) -> str: ...

    def evaluate_all(self, pair: SeriesPair, /) -> tuple[StatisticResult, ...]: ...
```

The adapter produces signed estimates and analytical validity records. Every
raw valid adapter result must have `selection_score is None`; a pre-scored raw
result is an integrity failure. The central selector is the only component that
writes selection scores. It returns `(selection, scored_results)`, and the
scored vector replaces the raw vector in `observed_results` and every
`ReplicateOutcome.statistic_results`.

Every observed and surrogate scan must return exactly one result for every
canonical candidate, in canonical order. Missing, duplicate, extra, or
reordered candidates are integrity failures, not analytical failures.

## 6. Executable null boundary

M0--M2 supports two explicit v2 null names:

```text
circular_shift_v2      {"min_shift": integer >= 1}
block_shuffle_v2       {"block_length": integer >= 1}
```

Both transform only the complete observed source series. The target remains
byte-equivalent. Common-support slicing occurs later inside the bound statistic
adapter. Transforming an already cropped support window is a different null
and is prohibited.

The unbound executable null exposes only:

```python
def bind(
    observed_pair: SeriesPair,
    *,
    semantic_input_sha256: str,
    scientific_plan_sha256: str,
) -> NullBindResult: ...
```

`NullBindResult` is a frozen sum type with exact invariants:

```python
class NullBindStatus(StrEnum):
    ENABLED = "enabled"
    DISABLED = "disabled"

class NullDisabledReason(StrEnum):
    SHIFT_SPACE_EMPTY = "shift_space_empty"
    NON_DIVISIBLE_TAIL = "non_divisible_tail"
    FEWER_THAN_TWO_BLOCKS = "fewer_than_two_blocks"
    TOO_MANY_BLOCKS = "too_many_blocks"

@dataclass(frozen=True, slots=True, eq=False)
class NullBindResult:
    status: NullBindStatus
    bound: BoundNullModel | None
    disabled_reason: NullDisabledReason | None
    diagnostics: tuple[str, ...]
```

`ENABLED` requires one exact bound object and no reason. `DISABLED` requires
no bound object and one enumerated reason. Disabled applicability is a normal
run-level `NOT_EVALUABLE` state, not a generic exception.

The bound null is attached to an immutable observed `SeriesPair`; callers
cannot pass a surrogate back into `apply` and cannot apply its tokens to a
different input.

```python
class BoundNullModel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, JsonValue]: ...

    @property
    def observed_length(self) -> int: ...

    @property
    def total_state_count(self) -> int: ...

    def identity_token(self) -> NullTransformToken: ...
    def sample_token(self, random: UniformIndexSource, /) -> NullTransformToken: ...
    def apply(self, token: NullTransformToken, /) -> NullTransformResult: ...
```

Each token is an exact frozen value object, not an arbitrary mapping:

```python
@dataclass(frozen=True, slots=True, eq=False)
class CircularShiftStateV2:
    schema: Literal["selcal.circular-shift-state.v2"]
    shift: int

@dataclass(frozen=True, slots=True, eq=False)
class BlockShuffleStateV2:
    schema: Literal["selcal.block-shuffle-state.v2"]
    block_order: tuple[int, ...]

NullStateV2 = CircularShiftStateV2 | BlockShuffleStateV2

@dataclass(frozen=True, slots=True, eq=False)
class NullTransformToken:
    schema: Literal["selcal.null-transform-token.v2"]
    null_name: str
    null_parameter_sha256: str
    semantic_input_sha256: str
    scientific_plan_sha256: str
    bound_null_owner_sha256: str
    is_identity: bool
    state: NullStateV2
```

The null-parameter digest is SHA-256 of these exact canonical JSON fields:

```json
{
  "schema": "selcal.null-parameters.v2",
  "name": "..._vN",
  "params": {}
}
```

The bound-owner digest is SHA-256 of these exact canonical JSON fields:

```json
{
  "schema": "selcal.bound-null-owner.v2",
  "semantic_input_sha256": "<lowercase sha256>",
  "scientific_plan_sha256": "<lowercase sha256>",
  "null_parameter_sha256": "<lowercase sha256>",
  "observed_length": 6,
  "transformed_role": "source"
}
```

The full token canonical payload uses the dataclass field names above and
rejects missing or extra fields. Its optional evidence digest is SHA-256 of
those canonical bytes. `apply` recomputes the null-parameter and owner digests,
checks all exact token fields, and verifies that `is_identity` agrees with the
state before touching data. A token from another bound null, observed input,
plan, or null contract raises an integrity exception.

`NullTransformResult` contains the new immutable pair, the exact token,
`source_changed`, and enumerated diagnostics.

The production Monte Carlo sampler draws uniformly with replacement from the
complete state space, including identity. A sampled identity is a valid
replicate, is retained in evidence, and must execute the same full statistic
scan and reselection. The observed identity remains the fixed additional anchor
in the plus-one formula. This combination follows the random-transformation
construction `(identity, g_1, ..., g_B)`, where every `g_b` is uniform on the
complete transformation set.

Different tokens that produce identical numerical arrays remain different
states with their original multiplicity. SelCal never deduplicates states by
transformed bytes and never resamples merely because the transformed source is
data-equivalent to the observed source. Such cases receive a diagnostic.

### 6.1 Circular shift v2

For source length `n` and `m = min_shift`, the nonidentity shift universe is

```text
K(n, m) = {m, m + 1, ..., n - m}.
```

The complete ordered state list is `(0, m, m+1, ..., n-m)`, with zero first.
`total_state_count = n - 2*m + 2`. Production sampling draws one index uniformly
from that list.

This is equivalent to requiring circular distance `min(k, n-k) >= m`.
Positive `k` is a right shift with `numpy.roll(source, k)` semantics, frozen
independently as `x_prime[j] = x[(j - k) mod n]`.

- `2*m > n`: the null is disabled with `SHIFT_SPACE_EMPTY`;
- `2*m == n`: one nonidentity state is allowed and an orbit-resolution
  diagnostic is recorded;
- zero is the identity state and may be sampled as a replicate;
- periodic data may make a nonzero shift data-equivalent; the token and its
  multiplicity are retained;
- for `m > 1`, the restricted set is not claimed to be a transformation group.
  Exact enumeration then proves implementation consistency only.

Canonical state payload embedded in the full owned token:

```json
{"schema": "selcal.circular-shift-state.v2", "shift": 2}
```

### 6.2 Block shuffle v2

The source is partitioned from index zero into `q = n // L` equal, complete,
labelled blocks of length `L = block_length`.

- `n % L != 0`: disabled with `NON_DIVISIBLE_TAIL`;
- `q < 2`: disabled with `FEWER_THAN_TWO_BLOCKS`;
- `q > 4096`: disabled with `TOO_MANY_BLOCKS` before constructing a block list
  or factorial-sized integer;
- no data are dropped, padded, fixed as an unreported tail, or mixed as an
  unequal short block;
- the complete state space contains all `q!` labelled block permutations,
  including identity order `(0, 1, ..., q-1)`;
- blocks keep their internal order and the source value multiset;
- repeated numerical blocks do not collapse labelled permutation states.

Canonical state payload embedded in the full owned token:

```json
{
  "schema": "selcal.block-shuffle-state.v2",
  "block_order": [2, 0, 1]
}
```

Production block sampling uses SelCal-owned Fisher--Yates over the ascending
label list `[0, ..., q-1]`. For `i = q-1, ..., 1`, draw
`j = random.randbelow(i+1)` and swap positions `i` and `j`. Identity is accepted;
there is no retry loop. The algorithm is uniform over all labelled
permutations and does not construct `q!` states. `total_state_count = q!` is
computed only after the `q <= 4096` guard and is never used to materialize the
state space.

## 7. Replicate seed and random-index contract

Each replicate has a replicate-local, deterministically domain-separated
stream. This means no shared mutable RNG state and no scheduling dependence;
it is not a claim of mathematical independence or collision impossibility.
Shared generators, `SeedSequence.spawn`, global `numpy.random`, Python
`hash()`, worker IDs, PIDs, wall time, task completion order, registry order,
and candidate order are forbidden inputs.

Inputs:

- scientific-plan digest: exactly 64 lowercase hexadecimal characters;
- replicate ID: exact built-in `int`, `0 <= id < B`, encoded as unsigned
  64-bit big-endian;
- stream name: the single ASCII literal `null_transform_v1` for this contract.

Preimage:

```text
b"SELCAL-SEED\x00\x01"
|| bytes.fromhex(scientific_plan_sha256)
|| replicate_id.to_bytes(8, "big", signed=False)
|| len(stream_ascii).to_bytes(2, "big", signed=False)
|| stream_ascii
```

Then:

```text
seed_digest = SHA256(preimage)                 # all 32 bytes
seed_uint256 = int.from_bytes(seed_digest, "big")
bitgen = numpy.random.PCG64(seed_uint256)
```

The canonical evidence field is the 64-character lowercase
`seed_digest_sha256`. A decimal integer representation is never the sole
persistent identity.

Only `PCG64.random_raw()` supplies random words. High-level `Generator`
methods such as `choice`, `integers`, `shuffle`, and `permutation` are outside
the reproducibility contract.

`randbelow` accepts only an exact built-in positive `int`. Uniform sampling
from an arbitrary positive bound is frozen as follows:

1. `bound == 1` returns zero without consuming a raw word;
2. let `k = (bound - 1).bit_length()` and `r = ceil(k / 64)`;
3. consume exactly `r` raw uint64 words per attempt, starting at `x=0` and
   updating `x = (x << 64) | int(word)` in returned order;
4. mask to the low `k` bits;
5. accept when the value is below `bound`, otherwise repeat.

This rejects modulo bias and supports block state spaces larger than `2**64`.

The primitive candidate known-answer vector for approval tests is:

```text
plan hash:     0000000000000000000000000000000000000000000000000000000000000000
replicate ID:  0
stream:        null_transform_v1
seed digest:   b48487717b221abeca05be2756c8439153a3002b10af07825cf1947f2e8db1d0
first raw64:   3260292358039148882
```

This vector remains a proposal until the document is approved and the value is
independently recalculated in tests.

The normative real-plan vectors derived from the v2 circular plan in Section 3
are:

```text
replicate 0 seed digest:
a8dd219a1e348a103ff10bdad3b9ae8d3cff53032be90a3603734530f51fb82a
first three raw64:
1874999576356411599, 4645183139923492514, 4210660991089764159

replicate 1 seed digest:
0fec37b5d56de9d84e86bb178f6715dcfb300db863201679e222a3e96240056e
first three raw64:
3445068325290577610, 519030488576969648, 7848502882682531391
```

For the plan's actual `n=6, min_shift=1`, the complete ordered circular states
are `(0,1,2,3,4,5)`; both streams sample shift `2` after applying the frozen
rejection rule. This repeated token across replicate IDs is legal.

The otherwise identical block-plan payload with
`null={"name":"block_shuffle_v2","params":{"block_length":2}}` has plan
digest `661d4b029d4f8dccc2545a8ad8776303ac06685e7410d48ffbe1484d41add810`.
For three labelled blocks its replicate-0 seed digest is
`2ab1b8f56e54245eda03bc438f79e00ee3c091b4367f255c1bc8818e6cfd59eb`,
and Fisher--Yates produces `(2,1,0)`. Replicate 1 produces `(1,0,2)`.

Using the Section 9 fixture pair but the Section 3 plan's candidates
`(1,2,3)`, the semantic-input digest is
`82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645`.
The circular null-parameter digest is
`a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3`
and its bound-owner digest is
`b5c33b348ba31477f0c085e7766eb39e1ce00b8a264fefa86eaf59d52c282831`.
The complete replicate-0 token is:

```json
{"bound_null_owner_sha256":"b5c33b348ba31477f0c085e7766eb39e1ce00b8a264fefa86eaf59d52c282831","is_identity":false,"null_name":"circular_shift_v2","null_parameter_sha256":"a2e6b608e05886b0c06388750e201c922c21d08e4ca6faf7668742798895dda3","schema":"selcal.null-transform-token.v2","scientific_plan_sha256":"dc7462598231451b36a94e2586037aa921a8159b2eabebb5fe5c81dcafa48931","semantic_input_sha256":"82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645","state":{"schema":"selcal.circular-shift-state.v2","shift":2}}
```

Its token digest is
`b5fadafeee3a831b3fd3b4db0ed3851274a74a4bdeff2201757c6e96ed983486`.

For the block plan, the null-parameter digest is
`7e59ac4c97edabc3549303e6035f25a8c262f378e5440fa6ee8aeac5ef37c28a`
and the bound-owner digest is
`90be6919fffddc18211b062ba0f913320bc997888a8968f895a2bb02ec7346e2`.
The complete replicate-0 token is:

```json
{"bound_null_owner_sha256":"90be6919fffddc18211b062ba0f913320bc997888a8968f895a2bb02ec7346e2","is_identity":false,"null_name":"block_shuffle_v2","null_parameter_sha256":"7e59ac4c97edabc3549303e6035f25a8c262f378e5440fa6ee8aeac5ef37c28a","schema":"selcal.null-transform-token.v2","scientific_plan_sha256":"661d4b029d4f8dccc2545a8ad8776303ac06685e7410d48ffbe1484d41add810","semantic_input_sha256":"82aa122557ed48908df8bb6ce226f05bdb5c7e4cf4287a671f74ee1eee9af645","state":{"block_order":[2,1,0],"schema":"selcal.block-shuffle-state.v2"}}
```

Its token digest is
`7584cdb3acb9b4dae6a49303fa3bc3f96764f91db43db2cb2e4deaa609430670`.

Boundary-vector tests use literal scripted words, not the production helper to
generate expected values:

| bound | scripted raw words | output | words consumed |
|---:|---|---:|---:|
| `1` | none | `0` | `0` |
| `2` | `3260292358039148882` | `0` | `1` |
| `3` | `3, 2` | `2` | `2` |
| `2**64` | `3260292358039148882` | `3260292358039148882` | `1` |
| `2**64 + 1` | `3260292358039148882, 11609078438225695210` | `11609078438225695210` | `2` |

The `bound=3` vector explicitly exercises one rejection.

The cross-version promise is deliberately narrow. For supported NumPy
`1.26.x` and `2.x` environments, CI must verify the same seed digests, PCG64
raw64 words, bounded indices, and transform tokens at the minimum-supported,
primary, and latest-supported dependency points. Pearson/NetTE floating-point
scores and complete calibration objects are not promised bitwise-identical
across NumPy versions. Exact NumPy and backend versions enter implementation
identity; cross-identity resume is an integrity failure.

## 8. Full-reselection calibration

The only M2 public entry point is conceptually:

```python
def calibrate_selected_family(
    pair: SeriesPair,
    resolution: PlanResolutionV2,
    /,
) -> CalibrationResult: ...
```

It accepts the exact sealed `PlanResolutionV2`. It does not accept a
`PlanRequestV2`, either v1 request/resolution type, a bare plan, independently
supplied statistic/null adapters,
candidate subset, worker configuration, failure-dropping policy, or successful
replicate denominator.

Data flow:

```text
immutable SeriesPair + sealed PlanResolutionV2
  -> reverify plan and adapter identities
  -> bind statistic once on observed data
  -> bind null once on the complete observed pair
  -> observed evaluate_all
  -> validate exact candidate vector
  -> central score/select; A_observed = true family maximum
  -> for replicate_id = 0, ..., B-1:
       derive replicate-local seed digest
       sample one complete-space transform with replacement, identity included
       apply token to the complete observed source
       evaluate_all on the surrogate pair
       validate exact candidate vector
       central score/select again
       retain complete outcome
  -> recompute E, failure count, p-value/bounds, and decision from outcomes
```

If every planned replicate is analytically valid,

```text
E = sum(A_b >= A_observed for b in 0..B-1)
p = (1 + E) / (B + 1)
reject_null = (p <= alpha)
```

The denominator is always the planned `B + 1`. Equality is an exceedance.
Candidate tie tolerance does not enter the tail comparison.

With finite complete-space sampling with replacement, this plus-one Monte
Carlo p-value is valid under the required transformation-group/invariance
conditions but is generally conservative. It is not the exact finite-space
enumeration p-value from Section 9.

A sampled identity is a normal COMPLETE replicate. Because its full scan is
the observed data under the same bound statistic, its decision statistic must
equal `A_observed` exactly within one implementation identity and therefore
counts as an exceedance. A mismatch is an integrity failure, not a tolerance
comparison.

## 9. Exact oracle

The V0 reference oracle enumerates mathematical transform states, not seeds or
observed random outputs. It is independent of the production sampler.

For `M` nonidentity states, it evaluates the observed identity state once and
every nonidentity state once, with a complete scan and reselection each time:

```text
p_exact = (1 + number of nonidentity states with A_state >= A_observed)
          / (M + 1)
```

The identity contributes the leading one. This exactly matches enumeration of
the defined finite state set. For circular `min_shift > 1`, it is an algorithm
oracle only because the restricted set is not a group. For a block shuffle, it
does not prove exchangeability of real nonstationary series.

Exact enumeration has `MAX_EXACT_STATES_V0 = 100_000`. State count is checked
incrementally before constructing tokens or permutations. Exceeding the cap
returns typed `EXACT_STATE_CAP_EXCEEDED`; it never silently switches to Monte
Carlo.

The reference oracle has an independent path: explicit integer shift loops for
circular states and `itertools.permutations(range(q))` for block states. It
does not call the production RNG, `randbelow`, circular sampler, Fisher--Yates,
or token sampler. Monkeypatching each production sampling helper to raise must
leave the reference-oracle tests green.

One normative circular/Pearson hand fixture is:

```text
source      = [0, 3, 1, 2, 4, 5]
target      = [0, 1, 3, 2, 5, 4]
candidates  = (1, 2)
selection   = max_upper
min_shift   = 1
states      = shifts (0, 1, 2, 3, 4, 5)
```

The full-scan reference table is:

| shift | scores `(lag 1, lag 2)` | selected lag | family max A |
|---:|---|---:|---:|
| 0 | `(0.4000000000000001, -0.4)` | 1 | `0.4000000000000001` |
| 1 | `(-0.4, 0.29111125486979084)` | 2 | `0.29111125486979084` |
| 2 | `(0.29111125486979084, -0.9561828874675149)` | 1 | `0.29111125486979084` |
| 3 | `(-0.9561828874675149, 0.05822225097395817)` | 2 | `0.05822225097395817` |
| 4 | `(0.05822225097395817, 0.7071067811865477)` | 2 | `0.7071067811865477` |
| 5 | `(0.7071067811865477, 0.4000000000000001)` | 1 | `0.7071067811865477` |

Three of six states meet or exceed the observed family maximum, so
`p_exact = 3/6 = 0.5`. This fixture is an implementation oracle, not evidence
that the particular series satisfy a shift-exchangeability null.

A separate scripted-adapter fixture freezes the anti-shortcut result:

```text
observed: (c1=3, c2=1) -> A=3, select c1
state 1:  (c1=2, c2=4) -> A=4, select c2
state 2:  (c1=3, c2=0) -> A=3, select c1
state 3:  (c1=1, c2=2) -> A=2, select c2
```

Complete reselection gives `3/4`; reusing observed `c1` gives `1/2` and must
fail the test.

The production random-transformation construction and its identity condition
must be checked against Phipson and Smyth's plus-one treatment and the group
conditions formalized by Hemerik and Goeman. SelCal does not promote a
restricted non-group transform set to an exact statistical test merely because
the algorithm matches this oracle.

## 10. Result objects and mathematical invariants

The v2 result contract must store enough information to validate the exact-B
mathematics:

```python
class RunFailureStage(StrEnum):
    NULL_BIND = "null_bind"
    OBSERVED_STATISTIC_SCAN = "observed_statistic_scan"
    REPLICATE_EXECUTION = "replicate_execution"

class ReplicateFailureStage(StrEnum):
    STATISTIC_SCAN = "statistic_scan"

@dataclass(frozen=True, slots=True, eq=False)
class ReplicateOutcome:
    replicate_id: int
    seed_digest_sha256: str
    status: ReplicateStatus
    failure_stage: ReplicateFailureStage | None
    transform_token: NullTransformToken
    statistic_results: tuple[StatisticResult, ...]
    selection: SelectionResult | None
    diagnostics: tuple[str, ...]

@dataclass(frozen=True, slots=True, eq=False)
class CalibrationResult:
    status: RunStatus
    failure_stage: RunFailureStage | None
    semantic_input_sha256: str
    scientific_plan_sha256: str
    planned_replicates: int
    alpha: float
    observed_results: tuple[StatisticResult, ...]
    observed_selection: SelectionResult | None
    replicates: tuple[ReplicateOutcome, ...]
    exceedance_count: int
    failure_count: int
    p_value: float | None
    exceedance_bound_low: float | None
    exceedance_bound_high: float | None
    reject_null: bool | None
    diagnostics: tuple[str, ...]
```

Validation is split by information ownership. Dataclass constructors reject
all self-contained contradictions visible in their stored fields. Exact
agreement with planned candidate IDs, selection rule, tie tolerance, adapter
identity, and `B` is checked by a public invariant verifier that requires the
exact sealed `PlanResolutionV2`; it is also called at calibration finalization.
The scientific plan is not duplicated into result fields merely to make a bare
constructor plan-aware. A result that has not passed this resolution-bound
verification is not an executable v2 calibration result.

For `COMPLETE`:

- `failure_stage is None`;
- replicate IDs are exactly `0..B-1`, unique and in canonical order;
- every outcome is complete and has a derivable seed and valid token;
- the observed vector and every replicate vector have candidate IDs exactly
  equal to planned `C`, all entries are VALID, all scores come from the central
  selector, and each selection index/candidate/tie set/decision statistic is
  consistent with its stored scored vector;
- `failure_count == 0`;
- `exceedance_count` is recomputed from stored
  `SelectionResult.decision_statistic` values;
- p-value is exact Python evaluation of `(1 + E) / (B + 1)`, with no
  approximate comparison, and reject is exactly `p_value <= alpha`;
- diagnostic bounds are `None`.

If `F > 0` replicates have analytical failure:

```text
low  = (1 + E) / (B + 1)
high = (1 + E + F) / (B + 1)
```

The final status is `NOT_EVALUABLE`; `p_value` and `reject_null` are `None`.
The bounds are diagnostics named missing-replicate exceedance bounds, not a
formal p-value interval and not a basis for a decision.

For replicate-failure `NOT_EVALUABLE`:

- `failure_stage is REPLICATE_EXECUTION`;
- `len(replicates) == B` and IDs are exactly ordered `0..B-1`;
- `F >= 1` and equals the number of failed outcomes;
- `E` is recomputed only from COMPLETE outcomes;
- `E + F <= B`;
- both bounds are present and exactly equal the formulas above;
- a statistic-scan failure retains the complete candidate vector and no
  selection; every replicate token is present.

Any missing planned outcome is an integrity/infrastructure state outside this
M2 result object, not a shorter `NOT_EVALUABLE` result.

If the observed scan has any analytical failure, the final result is
`NOT_EVALUABLE`, no null seed is consumed, no replicate is run, and p-value,
bounds, and reject decision are all `None`. `failure_stage` is
`OBSERVED_STATISTIC_SCAN`; the complete planned candidate vector is retained,
at least one entry is `ANALYTIC_FAILURE`, and selection is `None`.

If null binding is disabled, `failure_stage` is `NULL_BIND`, observed results
and replicates are empty, `exceedance_count == failure_count == 0`, and p-value,
bounds, and reject are `None`. Both hash fields are always exact lowercase
64-character SHA-256 strings.

`verify_calibration_result(result, resolution)` verifies self-contained result
arithmetic and invariants bound to the sealed resolution. Its two-argument
contract cannot authenticate `semantic_input_sha256` provenance for **any**
terminal status because it never receives the original pair. In COMPLETE and
REPLICATE_EXECUTION results, token-owner digests add representation and
cross-field consistency only: their inputs are public and can be recomputed for
a substituted semantic digest. In NULL_BIND and OBSERVED_STATISTIC_SCAN results,
no replicate token is present. Thus a bare verifier accepts only a claim about
representation, self-contained arithmetic, resolution consistency, and
token-owner consistency; it does not prove which real pair produced the digest.
This paragraph supersedes the earlier NULL_BIND-only claim ceiling.
`calibrate_selected_family(pair, resolution)` closes this boundary internally:
it freezes the entry pair, recomputes the semantic digest, checks it before the
single public verification call on every terminal path, invokes a
definition-time-frozen internal real verifier without exposing a second public
call, and rechecks the real semantic and plan digests immediately before return.
The internal verifier streams replicate exceedance accumulation and does not
materialize a second B-by-C result tree. No synthetic pair is fabricated and no
statistic or null adapter is rebound.

The integrity threat model is deliberately bounded. The contracts detect
ordinary replacement of canonical objects, stored fields, registry entries,
callbacks, and execution-time identities, including drift triggered through a
public verifier callback. Function-object `is` checks freeze ordinary callback
identity; they do not authenticate an interpreter against code that already has
arbitrary in-process write authority. In particular, no guarantee is made
against in-place mutation of a function's `__code__`, direct
`object.__setattr__` abuse, C-extension memory mutation, or equivalent host
interpreter compromise. Tests that inject field drift with `object.__setattr__`
exercise fail-closed detection of the resulting state; they are not a sandbox,
blind-validation result, or proof against arbitrary code execution.

## 10A. Versioned in-memory executor budget

The current reference in-memory executor applies a separate, versioned resource
envelope after successful null binding and observed-statistic evaluation:

```text
B <= 1000
B*C <= 5000
B*S <= 25000
B*(C*N + S + N) <= 2500000
```

Here `B` is the planned replicate count, `C` the canonical candidate count, `N`
the complete observed-series length, and `S` a generic retained-token state-unit
snapshot owned by the bound null. The circular snapshot reports `S=1`; the block
snapshot reports `S=q=N/block_length`. Generic calibration reads this frozen field
and contains no branch on a concrete null name. The four integer caps are the
rounded implementation-candidate envelope for the current reference environment;
they are not a promise of megabytes, elapsed seconds, throughput, or portability to
another allocator or platform.

Every cap is inclusive. For positive built-in integer operands no greater than
256 bits, the pure-integer gate computes `BC`, `BS`, and `work` in that fixed
order and rejects any overflow with `ResourceLimitError` whose stable message
begins `IN_MEMORY_EXECUTION_BUDGET_EXCEEDED_V2` and then reports
`B,C,N,S,BC,BS,work` followed by all four caps. An operand greater than 256 bits
is rejected before product or decimal materialization; its stable abbreviated
message reports `integer magnitude exceeds safe diagnostic envelope` and all
four caps without attempting to print the operand. The public calibrator and
the private replicate executor both enforce the same gate. A refusal occurs
before the `ReplicateRandomSource`, token sampling/application, block-label
list, replicate outcome, history, or signature paths, and it forms no
`CalibrationResult`, `RunFailureStage`, or diagnostic record.

Termination priority is unchanged: a disabled bind still returns at `NULL_BIND`,
and an analytical observed scan still returns at `OBSERVED_STATISTIC_SCAN`, before
this execution-only gate. Conversely, a successful observed scan is followed by
the gate before any B-scaled state is allocated. Plan resolution alone therefore
does not imply that the current executor can run the plan.

The 32-point pressure profile is the packaged
`in_memory_execution_budget_profile_20260828.csv`. It records unique point IDs,
the two supported nulls crossed with the two supported statistics, binned-NetTE
checks at `bins=32` and `bins=256`, and exact integer `S`, `BC`, `BS`, and `work`
columns. All 32 transcribed rows report `complete`; the stored integer columns
recompute exactly from `B`, `C`, `N`, and `S`.

Provenance: `COORDINATOR_TRANSCRIBED_FROM_SUBAGENT_TOOL_OUTPUT`. The profile was
reported for repository `HEAD=8f86c5a` on 2026-08-28, using an Apple M4 Pro,
macOS 26.6.2 arm64, CPython 3.12.10, and NumPy 2.2.6. The CSV is a transcribed
evidence table, not raw data. The original stdout and here-doc were not preserved
as independent artifacts. Each row is a single observation per point. There are
no repeats from which to estimate variance or a confidence interval.

Representative pressure points from the packaged profile are:

| ID | Null / statistic | B | C | N | S | bins | BC | BS | work | trace peak B | RSS delta B | wall s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cp_B1000_C1_N256` | circular / Pearson | 1000 | 1 | 256 | 1 | — | 1000 | 1000 | 513000 | 12123331 | 36519936 | 3.856852 |
| `cp_B100_C50_N128` | circular / Pearson | 100 | 50 | 128 | 1 | — | 5000 | 100 | 652900 | 16702984 | 52953088 | 1.896223 |
| `bn_joint_corner` | block / binned NetTE | 100 | 50 | 500 | 250 | 3 | 5000 | 25000 | 2575000 | 24234839 | 68468736 | 9.684666 |
| `bp_B20_C4_N4096_q4096` | block / Pearson | 20 | 4 | 4096 | 4096 | — | 80 | 81920 | 491520 | 29469678 | 96878592 | 2.681327 |
| `cn_B500_C5_N1024` | circular / binned NetTE | 500 | 5 | 1024 | 1 | 3 | 2500 | 500 | 3072500 | 12937934 | 38977536 | 8.288562 |
| `cn_B20_C4_N256_bins32` | circular / binned NetTE | 20 | 4 | 256 | 1 | 32 | 80 | 20 | 25620 | 2054427 | 6733824 | 0.257225 |
| `cn_B20_C4_N256_bins256` | circular / binned NetTE | 20 | 4 | 256 | 1 | 256 | 80 | 20 | 25620 | 2062685 | 6602752 | 0.254627 |

These baseline observations deliberately include points beyond the later candidate
caps, including `BS=81920` and `work=3072500`; those points now fail before replicate
allocation. They characterize the evidence considered when selecting a conservative
rounded envelope, not successful execution claims for the current guarded executor.
`trace_peak_B` is the Python allocation high-water reported by `tracemalloc`, which
does not include all native allocations. `rss_delta_B` is the reported difference
between process `ru_maxrss` high-water marks, not point-isolated retained memory or
an instantaneous RSS measurement.

Unknowns remain explicit. Allocator, OS, CPU, NumPy, workload values, and future
adapter implementations may change memory and timing. No independent upper envelope
for adjacent-bin/binned-NetTE bin-count regimes has been established; that question is
`HOLD` and is not falsely closed by this executor budget. Streaming, parallel,
persistent, UI, release, and scientific-validity claims remain outside this section.

## 11. Failure classification

SelCal catches only explicitly typed analytical failures. Generic
`RuntimeError`, `AssertionError`, I/O errors, cancellation, and programming
errors propagate or enter the later M3 infrastructure state machine; they are
not converted to ordinary statistical non-evaluability.

| Event | M2 action |
|---|---|
| null is disabled for the observed pair | `NOT_EVALUABLE`; zero seed consumption |
| observed candidate analytical failure | retain full vector; no selection or p-value |
| replicate candidate analytical failure | retain token and full vector; continue planned IDs |
| malformed candidate vector | integrity exception; stop |
| plan/adapter/bound identity drift | integrity exception; stop |
| sampled identity token | valid full-scan outcome; must equal observed A |
| data-equivalent nonidentity token | valid outcome plus diagnostic; do not resample |
| retry after later infrastructure failure | same ID, seed, and token |

For the two v2 finite nulls, no legitimate post-bind null analytical failure is
defined. Empty state spaces are handled at bind. Out-of-range random indices,
malformed or foreign tokens, target drift, shape drift, value-multiset drift,
and impossible application failures are integrity errors and stop M2; they are
never counted in `F`.

Identity verification occurs at fixed points:

1. entry: rehash the exact v2 plan and verify unbound statistic/null names and
   parameters against the sealed resolution;
2. post-bind: verify bound names, parameters, candidates, semantic-input hash,
   plan hash, and owner digest;
3. every scan: verify every result's backend and preprocessing identity against
   the bound snapshot and verify raw scores are absent;
4. every replicate before and after transformation/scan: verify plan, adapter,
   bound, token-owner, input-role, and target identities;
5. finalization: rehash the plan, rederive every seed, validate every token, and
   reconstruct all counts and decisions from stored outcomes.

Any mismatch raises a typed integrity exception and produces no
`NOT_EVALUABLE` result.

Failed replicates are never deleted, replaced, renumbered, or removed from the
denominator. M2 is an in-memory scientific core and writes no event log,
resume state, manifest, evidence bundle, or report.

## 12. Migration and compatibility

1. Characterization tests freeze the existing v1 payload bytes, v1 hashes, and
   current representative-score behavior.
2. V1 remains parseable and hashable only through
   `scientific_plan_v1_sha256(ResolvedScientificPlan)`.
3. V2 is hashed only through
   `scientific_plan_v2_sha256(ResolvedScientificPlanV2)`; both functions require
   exact types.
4. V1 is rejected by the executable v2 calibrator with `PlanVersionError`.
5. `migrate_plan_v1_to_v2(request: PlanRequest) -> PlanMigrationV1ToV2` is the
   sole migration entry point. It maps `circular_shift_v1` to
   `circular_shift_v2` or `block_shuffle_v1` to `block_shuffle_v2`, resolves the
   new request through the v2 registry, and returns both exact plan digests plus
   a canonical field-by-field change ledger.
6. Migration never claims semantic equivalence, never overwrites the v1 object,
   and never submits the migrated v2 object for execution implicitly.
7. A v1 root seed above `2**64 - 1`, replicate count above `1_000_000`, or any
   value invalid under v2 causes a typed migration refusal recorded in the
   ledger; values are never clipped, rounded, or defaulted.

## 13. Adversarial acceptance tests

Before M0--M2 can be called locally complete, tests must cover at least:

1. v1 golden payload and hash preservation;
2. v1 rejection at the v2 execution boundary;
3. true family maximum versus within-tolerance representative score;
4. signed estimate preservation under `max_absolute`;
5. full candidate scan and reselection in every surrogate;
6. the hand oracle where full reselection gives `3/4` but reusing the observed
   candidate gives `1/2`;
7. inclusive equality and plus-one correction;
8. circular direction, symmetric distance set, empty and one-state spaces;
9. strict block divisibility, identity inclusion, frozen Fisher--Yates mapping,
   repeated numerical blocks, and block-count cap;
10. complete-source transform before common-support slicing;
11. exact v2 payload field set, canonical bytes/hash, root seed, ID, stream,
    endian, raw64, bounded-index, circular-token, and block-token vectors;
12. order independence across candidates, registries, workers, retries, and
    process scheduling;
13. supported NumPy versions for PCG64 raw words and SelCal's own bounded-index
    mapping;
14. `B > state_count` with legal repeated tokens and sampled identities under
    complete-space sampling with replacement;
15. duplicate, missing, out-of-range, and unordered replicate IDs;
16. seed, token, plan, adapter, and backend identity drift;
17. observation failure after one successful null bind, with zero RNG,
    token-sampling, token-application, or replicate calls;
18. failed replicate retention and exact diagnostic bounds;
19. immutability and absence of writable input aliases;
20. exact oracle independence from the production RNG and sampler.
21. token rejection across another input, plan, null parameterization, or bound
    owner;
22. run-level failure stages and exact constructor invariants for null-bind,
    observation, and replicate failures;
23. raw adapter scores rejected and only central scored vectors persisted;
24. exact-state and execution-resource caps fail before state materialization or
    replicate allocation.

Tests must also demonstrate that core modules do not branch on concrete
statistic or null names outside registries/factories.

## 14. Implementation sequence and gates

After written approval, implementation proceeds by dependency-safe,
test-driven slices:

1. v1 characterization, the unresolved `PlanRequestV2`, self-contained v2
   result/state/token value contracts, and structural protocols;
2. central family-max selection and the raw `evaluate_all` protocol;
3. seed framing, PCG64 raw-word wrapper, and uniform-index mapping;
4. deterministic circular state enumeration, owned tokens, and application;
5. deterministic equal-block application and frozen Fisher--Yates sampling;
6. the resolver-only sealed v2 plan, exact v2 payload/hash, and explicit
   migration ledger. Earlier token/RNG known-answer tests use only the approved
   literal plan digest and never construct or impersonate a resolved plan;
7. null bind, observed scan, and fail-closed observation path;
8. exact-B replicate execution with full reselection;
9. p-value, failure bounds, and resolution-bound invariant reconstruction;
10. independent exact-oracle fixtures and adversarial identity tests;
11. public-boundary and filter-net evidence mapping;
12. full branch coverage, Ruff, strict mypy, build, clean install,
    cross-process, supported-version matrix, and evidence-only status closure.

Stop if any slice requires a shared RNG, worker-dependent seed, Python hash,
silent v1 reinterpretation, selected-candidate-only surrogate evaluation,
short-tail block handling, output-based transform deduplication, failed-
replicate deletion, nonidentity-only with-replacement sampling, or a
successful-replicate denominator.

Passing this sequence permits the label
`M0-M2 IN-MEMORY CONTRACT LOCALLY VERIFIED`. It does not permit
`SCIENTIFICALLY CALIBRATED`, `SOFTWAREX READY`, `A+`, `MINOR REVISION LIKELY`,
or `UPLOAD READY`.

## 15. M6 and downstream boundary

M6 remains the next scientific gate after M0--M2. It must separately test:

- exact finite-state agreement;
- strict null calibration under eligible data-generating mechanisms;
- power and selection behavior under alternatives;
- common-driver, bidirectional, instantaneous-mixing, and feedback stress
  cases without mislabelling them as strict nulls;
- null-assumption violations and analytical failure envelopes;
- multi-statistic invariance of the orchestration contract;
- comparison against named alternatives under matched support and estimands;
- real-data usability with claim ceilings and adverse results retained.

UI, persistence, evidence bundles, release engineering, public repository,
license choice, DOI deposit, and the SoftwareX manuscript remain downstream.
No UI or manuscript work starts before the M6 result and its claim ceiling are
frozen.

## 16. Filter-net writeback candidates

The following project-specific rules are proposed for activation with this
design:

1. A plan hash containing an undefined RNG label is not a reproducibility
   contract.
2. A tolerance-selected representative is not automatically the family maximum;
   selection labels and inferential statistics must be separate fields.
3. Exact-oracle enumeration counts mathematical transform states, including
   multiplicity, not unique output arrays and not sampled seeds.
4. A short block is a scientific null-definition decision, not an implementation
   edge case.
5. Cross-version reproducibility may rely on a documented raw bit-generator
   stream plus SelCal-owned mapping, not an unfrozen high-level sampler.
6. Complete replicate identity means exactly `0..B-1`; a successful subset
   cannot support the planned denominator.
7. Algorithmic exactness does not establish exchangeability or Type-I-error
   control; those remain M6 evidence questions.
8. Nonidentity-only sampling with replacement plus an observed `+1` anchor can
   target the wrong distribution; the random replicate space and exact-oracle
   space must be the same complete transform set.

These rules are not yet written into the portfolio-wide permanent filter net.
They become eligible only after written design approval and implementation
evidence demonstrates that the proposed controls are practical.

## 17. Primary references for the frozen boundary

- Phipson, B. and Smyth, G. K. (2010),
  [Permutation p-values should never be zero](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).
- Hemerik, J. and Goeman, J. (2018),
  [Exact testing with random permutations](https://pmc.ncbi.nlm.nih.gov/articles/PMC6405018/).
- NumPy,
  [PCG64 compatibility guarantee](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html)
  and [random-stream compatibility policy](https://numpy.org/doc/2.0/reference/random/compatibility.html).
