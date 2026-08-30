"""Tests for the multi-agent layer.

Every agent has a deterministic fallback, so the whole team is testable without an
API key and without spending tokens. The LLM paths are exercised by feeding
hand-built payloads through `_parse`, which is where validation lives.
"""
import pytest

from agents.context import (DIMENSIONS, ResearchContext, TrialRecord,
                            command_signature)
from agents.engineer import EngineerAgent, validate_args
from agents.product_manager import Directive, ProductManagerAgent
from agents.qa import QAAgent
from agents.researcher import Hypothesis, ResearchAgent
from agents.team import AgentTeam
from orchestrator.schemas import TokenUsage


def ctx(**kw):
    base = dict(baseline=0.6016, max_iterations=50, wall_clock_budget_s=21600)
    base.update(kw)
    return ResearchContext(**base)


def trial(i, dim, primary, cmd="--model fm_torch --loss listwise"):
    return TrialRecord(iteration=i, dimension=dim, hypothesis="h", command=cmd,
                       signature=command_signature(cmd), primary=primary,
                       status="ACCEPTED" if primary else "FAILED")


# --- duplicate detection ---------------------------------------------------

def test_signature_ignores_ordering_interpreter_and_data_dir():
    a = "/usr/bin/python -m pipeline.train --model mmoe --experts 6 --data_dir /x"
    b = "python3 -m pipeline.train --experts 6 --model mmoe"
    assert command_signature(a) == command_signature(b)


def test_signature_distinguishes_real_differences():
    assert (command_signature("--model mmoe --experts 4")
            != command_signature("--model mmoe --experts 6"))


def test_duplicate_configuration_is_detected():
    """Regression: an LLM run re-ran an identical config while claiming it changed one.

    That wasted an iteration and helped trigger the convergence halt.
    """
    c = ctx()
    c.record(trial(1, "multi_task", 0.6039, "--model mmoe --experts 6 --aux_weight 0.3"))
    assert c.is_duplicate("--model mmoe --aux_weight 0.3 --experts 6")
    assert not c.is_duplicate("--model mmoe --experts 8 --aux_weight 0.3")


# --- product manager: coverage --------------------------------------------

def test_pm_directs_at_an_untried_dimension_first():
    """The failure this role exists to prevent: never trying the loss axis."""
    c = ctx()
    for i in range(1, 5):
        c.record(trial(i, "architecture", 0.60 + i / 1000))
    d = ProductManagerAgent().fallback(c)
    assert "architecture" not in d.focus_dimensions
    assert set(d.focus_dimensions) <= set(c.unexplored_dimensions())


def test_pm_prefers_loss_when_everything_is_untried():
    assert ProductManagerAgent().fallback(ctx()).focus_dimensions[0] == "loss"


def test_pm_rotates_away_when_progress_stalls():
    c = ctx()
    c.record(trial(1, "loss", 0.6024))
    for i, d in enumerate(DIMENSIONS, start=2):      # touch every dimension
        c.record(trial(i, d, 0.5990))
    d = ProductManagerAgent().fallback(c)
    assert "loss" in d.avoid_dimensions or "loss" not in d.focus_dimensions


def test_pm_refreshes_on_schedule_not_every_iteration():
    pm = ProductManagerAgent(refresh_every=3)
    c = ctx()
    c.iteration = 1
    assert pm.should_refresh(c)             # no directive yet
    c.directive = pm.fallback(c)
    c.iteration = 2
    assert not pm.should_refresh(c)         # cost control: not every iteration
    c.iteration = 4
    assert pm.should_refresh(c)             # 4 % 3 == 1


def test_directive_rejects_invented_dimensions():
    with pytest.raises(Exception):
        Directive(phase="p", focus_dimensions=["quantum_entanglement"])


def test_directive_filters_unknown_but_keeps_valid():
    d = Directive(phase="p", focus_dimensions=["loss", "not_a_dimension"])
    assert d.focus_dimensions == ["loss"]


# --- researcher ------------------------------------------------------------

def test_researcher_stays_inside_the_directive():
    c = ctx()
    c.directive = Directive(phase="p", focus_dimensions=["loss"])
    hyps = ResearchAgent(proposals=3).fallback(c)
    assert hyps and all(h.dimension == "loss" for h in hyps)


def test_researcher_parse_drops_out_of_scope_hypotheses():
    c = ctx()
    c.directive = Directive(phase="p", focus_dimensions=["loss"])
    payload = {"hypotheses": [
        {"dimension": "loss", "hypothesis": "try listwise softmax", "args": "--model fm_torch --loss listwise"},
        {"dimension": "capacity", "hypothesis": "make embeddings bigger", "args": "--model fm_torch --embed_dim 64"},
    ]}
    kept = ResearchAgent()._parse(payload, c)
    assert [h.dimension for h in kept] == ["loss"]


def test_hypothesis_rejects_unknown_dimension():
    with pytest.raises(Exception):
        Hypothesis(dimension="vibes", hypothesis="something long enough", args="--model fm")


# --- engineer: validation is the anti-drift guarantee ----------------------

def test_engineer_rejects_arguments_the_trainer_does_not_accept():
    c = ctx()
    h = Hypothesis(dimension="loss", hypothesis="test an invented flag",
                   args="--model fm_torch --lambdarank_truncation 5")
    assert EngineerAgent().build(c, [h]) is None


def test_engineer_falls_through_to_the_next_hypothesis():
    c = ctx()
    bad = Hypothesis(dimension="loss", hypothesis="invalid model choice", args="--model dcnv2")
    good = Hypothesis(dimension="loss", hypothesis="within-user listwise softmax",
                      args="--model fm_torch --loss listwise")
    spec = EngineerAgent().build(c, [bad, good])
    assert spec is not None and spec.args == "--model fm_torch --loss listwise"
    assert spec.rejected, "the rejected candidate must be recorded for the log"


def test_engineer_turns_a_duplicate_into_a_seed_replicate():
    """A repeat under a new seed measures the noise floor; a repeat under the same seed is waste."""
    c = ctx()
    c.record(trial(1, "loss", 0.6024, "--model fm_torch --loss listwise"))
    h = Hypothesis(dimension="loss", hypothesis="within-user listwise softmax",
                   args="--model fm_torch --loss listwise")
    spec = EngineerAgent().build(c, [h])
    assert spec is not None and "--seed" in spec.args


def test_engineer_binds_hypothesis_to_command():
    """The pair must travel together — the drift bug produced a log that claimed
    DCN-v2 while running --model mmoe."""
    c = ctx()
    h = Hypothesis(dimension="multi_task", hypothesis="MMoE with auxiliary click and like heads",
                   args="--model mmoe --loss listwise")
    spec = EngineerAgent().build(c, [h])
    assert spec.hypothesis == h.hypothesis
    assert "--model mmoe" in spec.command
    assert spec.checkpoint == "mmoe"


def test_engineer_threads_data_dir():
    spec = EngineerAgent().build(
        ctx(), [Hypothesis(dimension="loss", hypothesis="listwise softmax test",
                           args="--model fm_torch --loss listwise")],
        data_dir="data/KuaiRand-Pure/data")
    assert "--data_dir data/KuaiRand-Pure/data" in spec.command


# --- QA --------------------------------------------------------------------

def test_qa_blocks_a_hypothesis_that_contradicts_its_command():
    c = ctx()
    from agents.engineer import TrialSpec
    spec = TrialSpec(dimension="architecture",
                     hypothesis="Deep & Cross Network v2 captures explicit feature crosses",
                     args="--model mmoe --loss listwise",
                     command="python -m pipeline.train --model mmoe --loss listwise",
                     checkpoint="mmoe")
    verdict = QAAgent().preflight(c, spec)
    assert not verdict.ok
    assert any("does not mention" in p for p in verdict.problems)


def test_qa_rejects_scores_below_the_random_floor():
    v = QAAgent().judge(0.40)
    assert not v.trustworthy and "random" in v.note


def test_qa_rejects_scores_above_the_oracle_ceiling():
    """Beating the oracle is only possible with label leakage."""
    v = QAAgent().judge(0.95)
    assert not v.trustworthy and "leakage" in v.note


def test_qa_accepts_a_plausible_score():
    assert QAAgent().judge(0.6039).trustworthy


def test_qa_flags_a_weak_but_real_result():
    v = QAAgent().judge(0.5500)
    assert v.trustworthy and "popularity" in v.note


# --- team integration ------------------------------------------------------

def test_team_plans_an_iteration_without_an_llm():
    team = AgentTeam(TokenUsage(), verbose=False)
    c = ctx()
    c.iteration = 1
    plan = team.plan(c)
    assert plan.ok
    assert plan.directive is not None
    assert plan.spec.dimension in DIMENSIONS
    ok, _ = validate_args(plan.spec.args)
    assert ok


def test_team_never_repeats_a_configuration_across_iterations():
    """The end-to-end property the whole design exists to guarantee."""
    team = AgentTeam(TokenUsage(), verbose=False)
    c = ctx()
    seen = set()
    for i in range(1, 9):
        c.iteration = i
        plan = team.plan(c)
        if not plan.ok:
            continue
        sig = command_signature(plan.spec.args)
        assert sig not in seen, f"iteration {i} repeated {sig}"
        seen.add(sig)
        team.record(c, i, plan.spec, 0.60 + i / 10000, "ACCEPTED")
    assert len(seen) >= 6


def test_team_covers_multiple_dimensions_over_a_run():
    """The other end-to-end property: no more four-iterations-of-one-axis runs."""
    team = AgentTeam(TokenUsage(), verbose=False)
    c = ctx()
    for i in range(1, 7):
        c.iteration = i
        plan = team.plan(c)
        if plan.ok:
            team.record(c, i, plan.spec, 0.60, "REJECTED")
    touched = {t.dimension for t in c.trials}
    assert len(touched) >= 3, f"only explored {touched}"


def test_team_costs_nothing_without_an_llm():
    meter = TokenUsage()
    team = AgentTeam(meter, verbose=False)
    c = ctx()
    c.iteration = 1
    team.plan(c)
    assert meter.total == 0 and meter.calls == 0


# --- flag spelling ---------------------------------------------------------

def test_equals_and_space_flag_forms_are_equivalent():
    """argparse accepts both; so must everything that reads a command.

    Regression: a real LLM run emitted `--model=deepfm`, which the checkpoint
    parser did not understand, so QA blocked two iterations for "no --model" on
    commands that plainly had one.
    """
    from agents.engineer import checkpoint_name
    from sandbox.logger import RunLogger
    assert checkpoint_name("--model=deepfm --loss=listwise") == "deepfm_listwise"
    assert checkpoint_name("--model deepfm --loss listwise") == "deepfm_listwise"
    assert (command_signature("--model=mmoe --experts=6")
            == command_signature("--model mmoe --experts 6"))
    assert RunLogger._infer_checkpoint("python -m pipeline.train --model=mmoe") == "mmoe"


def test_qa_preflight_accepts_equals_form():
    from agents.engineer import TrialSpec
    spec = TrialSpec(dimension="architecture",
                     hypothesis="DeepFM adds an MLP branch over the field embeddings",
                     args="--model=deepfm --loss=listwise",
                     command="python -m pipeline.train --model=deepfm --loss=listwise",
                     checkpoint="deepfm_listwise")
    assert QAAgent().preflight(ctx(), spec).ok


# --- QA coherence check is dimension-aware ---------------------------------

def _spec(dimension, hypothesis, args):
    from agents.engineer import TrialSpec, checkpoint_name
    return TrialSpec(dimension=dimension, hypothesis=hypothesis, args=args,
                     command=f"python -m pipeline.train {args}",
                     checkpoint=checkpoint_name(args))


def test_qa_accepts_a_loss_hypothesis_that_does_not_name_the_model():
    """Regression: a real run lost an iteration here.

    The Researcher proposed a BPR experiment that happened to run on DeepFM. A
    model-only coherence check blocked it for not saying "DeepFM", even though the
    hypothesis was about the objective and the architecture was incidental.
    """
    spec = _spec("loss",
                 "Compare a pairwise BPR objective against listwise ranking.",
                 "--model=deepfm --loss=bpr")
    assert QAAgent().preflight(ctx(), spec).ok


def test_qa_still_blocks_a_genuine_architecture_mismatch():
    spec = _spec("architecture",
                 "Deep & Cross Network v2 captures explicit feature crosses",
                 "--model=mmoe --loss=listwise")
    v = QAAgent().preflight(ctx(), spec)
    assert not v.ok and any("does not mention" in p for p in v.problems)


def test_qa_blocks_a_loss_hypothesis_naming_the_wrong_objective():
    spec = _spec("loss",
                 "Test the listwise softmax objective for within-user ranking.",
                 "--model=fm_torch --loss=bpr")
    assert not QAAgent().preflight(ctx(), spec).ok
