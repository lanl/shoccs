export const meta = {
  name: 'plan-48-cutcell-normal-modes',
  description: 'Execute plan 48: generalized Kreiss, reflection coefficients, round-trip tier, RK4 containment + L8 upgrades',
  whenToUse: 'Run with the Workflow tool from the repo root to execute plans/48-cutcell-normal-modes.md item by item.',
  phases: [
    { title: 'Phase 0', detail: 'Substrate hardening: defective-kappa observability, augmented Kreiss, all-rows L2, near-origin treatment' },
    { title: 'Phase 1', detail: 'Cut-cell ingestion: cached evaluator, psi Kreiss check, psi margin, calibration' },
    { title: 'Phase 2', detail: 'Reflection coefficients: mode split, R(s), mirror, psi sweep + report, Gaussian calibration' },
    { title: 'Phase 3', detail: 'Two-boundary tier: interval assembler, resolvent scan, round-trip, wiring + docs' },
    { title: 'Phase 4', detail: 'RK4 containment (resolvent-only) + tgb fix + L8 cut-cell upgrades + acceptance' },
  ],
}

const PLAN = 'plans/48-cutcell-normal-modes.md'

// One entry per plan item, in dependency order. `parallelOk` items within the
// same group run concurrently (they share no files). `kind: 'findings'` items
// produce a Findings-log append instead of production code.
const ITEMS = [
  { id: '0.1', phase: 'Phase 0', title: 'Defective-kappa observability', tests: 'tests/test_gks_kreiss.py' },
  { id: '0.2', phase: 'Phase 0', title: 'Augmented Kreiss matrix (m >= p rows)', tests: 'tests/test_gks_kreiss.py' },
  { id: '0.3', phase: 'Phase 0', title: 'Feed all differing closure rows to L2', tests: 'tests/test_gks_kreiss.py tests/test_brady2d_stability.py' },
  { id: '0.4', phase: 'Phase 0', title: 'Near-origin treatment replaces disk mask', tests: 'tests/test_gks_kreiss.py' },
  { id: '1.1', phase: 'Phase 1', title: 'Cached lambdified cut-cell evaluator', tests: 'tests/test_cutcell_eval.py' },
  // 0.5 was added after 1.2's first attempt blocked on gks_kreiss.py substrate defects.
  // It is placed AFTER 1.1 in execution order so the resume cache prefix (0.1..1.1) replays.
  { id: '0.5', phase: 'Phase 0', title: 'Witness-search robustness for augmented Kreiss', tests: 'tests/test_gks_kreiss.py tests/test_cutcell_eval.py' },
  { id: '1.2', phase: 'Phase 1', title: 'psi-parameterized Kreiss check (after 0.5 substrate fixes)', tests: 'tests/test_cutcell_eval.py' },
  { id: '1.3', phase: 'Phase 1', title: 'Worst-case-over-psi margin', tests: 'tests/test_cutcell_eval.py -k "psi_sweep or margin"' },
  { id: '1.4', phase: 'Phase 1', title: 'Calibration / control experiment', tests: null, kind: 'findings' },
  { id: '2.1', phase: 'Phase 2', title: 'Mode partition', tests: 'tests/test_reflection.py -k mode_split' },
  { id: '2.2', phase: 'Phase 2', title: 'Reflection matrix and margin', tests: 'tests/test_reflection.py' },
  { id: '2.3', phase: 'Phase 2', title: 'Right-end closures (mirror convention)', tests: 'tests/test_reflection.py -k mirror' },
  { id: '2.4', phase: 'Phase 2', title: 'psi-swept reflection + layer_cutcell wiring', tests: 'tests/test_reflection.py tests/test_optimizer.py::TestExtractField' },
  { id: '2.5', phase: 'Phase 2', title: 'Gaussian eps=0.1 mechanism calibration', tests: 'tests/test_reflection.py -k gaussian' },
  { id: '3.1', phase: 'Phase 3', title: 'Interval-operator assembler', tests: 'tests/test_interval_operator.py' },
  { id: '3.2', phase: 'Phase 3', title: 'Resolvent-based interval scan', tests: 'tests/test_interval_operator.py -k scan' },
  { id: '3.3', phase: 'Phase 3', title: 'Round-trip criterion (Trefethen 1985 Lemma 1)', tests: 'tests/test_reflection.py -k round_trip' },
  { id: '3.4', phase: 'Phase 3', title: 'layer_cutcell round-trip wiring + docs + skill update', tests: 'tests/ -k "cutcell or reflection or interval"' },
  { id: '4.1', phase: 'Phase 4', title: 'RK4 boundary containment (resolvent only)', tests: 'tests/test_fully_discrete.py' },
  { id: '4.2', phase: 'Phase 4', title: 'transient_growth_bound honesty fix', tests: 'tests/test_non_normality.py tests/test_brady2d_stability.py' },
  { id: '4.3', phase: 'Phase 4', title: 'L8 cut-cell capability investigation gate', tests: null, kind: 'findings' },
  { id: '4.4', phase: 'Phase 4', title: 'L8 upgrades: cut template, growth rate, dispatch, CFL', tests: 'tests/test_cpp_bridge.py' },
  { id: '4.5', phase: 'Phase 4', title: 'End-to-end acceptance comparison', tests: null, kind: 'findings' },
]

const IMPL_SCHEMA = {
  type: 'object',
  required: ['status', 'summary', 'files_changed', 'test_output_tail'],
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    test_output_tail: { type: 'string', description: 'last ~20 lines of the focused test run (or findings summary)' },
    blocked_reason: { type: 'string' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['approved', 'summary'],
  properties: {
    approved: { type: 'boolean' },
    summary: { type: 'string' },
    issues: { type: 'array', items: { type: 'string' }, description: 'concrete defects with file:line; empty if approved' },
  },
}

function implPrompt(item, feedback) {
  return [
    'You are implementing one item of an approved implementation plan in /workspace (SHOCCS; SymPy pipeline in scripts/stencil_gen).',
    'PLAN: read ' + PLAN + ' in full — the header, constraints, Read-first list, and especially item ' + item.id + ' ("' + item.title + '"). The plan is the spec; follow it exactly, including the math as written (except where the plan itself tells you to verify against the cited paper).',
    'HARD CONSTRAINTS: no time integration / matrix powers / expm in Python (resolvent and root-locus computations only); the Dirichlet (wall-eliminated) block is the canonical analysis object; mark E4_1 symbolic-derivation tests @pytest.mark.slow; test conventions per the plan.',
    item.kind === 'findings'
      ? 'THIS IS A FINDINGS ITEM: run the prescribed experiment/investigation, then APPEND a dated subsection to the "## Findings log" section of ' + PLAN + ' with the numbers/evidence. No production code. If the item defines an acceptance criterion, state pass/fail explicitly.'
      : 'Implement the item (one concept; touch only the files the item names), add the prescribed tests, then run the focused test command from scripts/stencil_gen: SYMPY_CACHE_SIZE=50000 uv run pytest ' + item.tests + ' -x -q. Iterate until green.',
    'When done, commit ONLY this item\'s changes with message "plan48 ' + item.id + ': ' + item.title + '". If genuinely blocked (missing precondition, plan contradiction), do not commit — return status blocked with a precise reason.',
    feedback ? '\nREVIEWER FEEDBACK ON YOUR PREVIOUS ATTEMPT (fix all of these):\n- ' + feedback.join('\n- ') : '',
    'Your final output is structured data for the driver, not prose.',
  ].join('\n\n')
}

function reviewPrompt(item, impl) {
  const testStep = item.tests
    ? '(4) run the focused tests yourself from scripts/stencil_gen: SYMPY_CACHE_SIZE=50000 uv run pytest ' + item.tests + ' -x -q.'
    : '(4) no test command — this is a findings item; verify the Findings log entry in ' + PLAN + ' instead.'
  return [
    'You are the adversarial reviewer for one implemented item of ' + PLAN + ' in /workspace. Item ' + item.id + ' ("' + item.title + '"). Implementer report: ' + impl.summary + '. Files: ' + impl.files_changed.join(', ') + '.',
    'Read the plan item and the actual diff (git show HEAD; git diff HEAD~1 if committed). Verify: (1) MATH — re-derive independently any formula the item specifies (augmented Kreiss columns, reflection RHS b_k, mirror signs, Lemma 1 ordering against Trefethen 1985 if this is item 3.3); (2) SPEC — every prescribed test exists and asserts what the plan says (not weakened tolerances); (3) CONSTRAINTS — no time-stepping/matrix powers in Python paths; slow markers present on E4_1 derivations; ' + testStep,
    'For findings items: check the Findings log entry exists, contains actual numbers, and the acceptance criterion is honestly evaluated.',
    'Approve ONLY if all four checks pass. Issues must be concrete (file:line). Your final output is structured data.',
  ].join('\n\n')
}

async function runItem(item) {
  let feedback = null
  for (let attempt = 1; attempt <= 2; attempt++) {
    const impl = await agent(implPrompt(item, feedback), {
      label: 'impl:' + item.id + (attempt > 1 ? ':retry' : ''),
      phase: item.phase,
      schema: IMPL_SCHEMA,
    })
    if (!impl) return { item: item.id, ok: false, reason: 'implementer agent failed' }
    if (impl.status === 'blocked') return { item: item.id, ok: false, reason: impl.blocked_reason || impl.summary }

    const review = await agent(reviewPrompt(item, impl), {
      label: 'review:' + item.id + (attempt > 1 ? ':retry' : ''),
      phase: item.phase,
      schema: REVIEW_SCHEMA,
    })
    if (!review) return { item: item.id, ok: false, reason: 'reviewer agent failed' }
    if (review.approved) return { item: item.id, ok: true, summary: impl.summary, review: review.summary }
    feedback = review.issues && review.issues.length ? review.issues : [review.summary]
    log('Item ' + item.id + ' rejected (attempt ' + attempt + '): ' + feedback.join(' | '))
  }
  return { item: item.id, ok: false, reason: 'rejected twice by reviewer: ' + feedback.join(' | ') }
}

// Strictly sequential: every item commits before the next starts (agents share
// one worktree; concurrent commits would race on the git index).
const results = []
for (const item of ITEMS) {
  phase(item.phase)
  log('Item ' + item.id + ': ' + item.title)
  const r = await runItem(item)
  results.push(r)
  if (!r.ok) { log('BLOCKED at ' + r.item + ': ' + r.reason); return { completed: results, blocked_at: r.item, reason: r.reason } }
}

log('All ' + results.length + ' items complete')
return { completed: results, blocked_at: null }
