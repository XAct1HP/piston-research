# Piston System Research Tool

A parametric design and analysis tool for the reciprocating load chain of a
combustion engine: cylinder sleeve, piston, rings, gudgeon pin, small end and
connecting rod. The goal is to redesign those parts to be lighter, stronger and
able to carry more power, with every number traceable to an equation you can
look up.

The full architecture and the seven-phase build plan live in the project's
build plan document. This README covers what exists today and how to run it.

## Status: all seven phases built

Give it an engine's specifications and it returns the complete load environment
its piston system lives in, the temperature every region runs at, and whether
each of nine components survives -- with the equation, the reference and every
input behind each answer. Since phase 3 the masses driving all of that are
measured off real parametric solids rather than typed in, and those solids
export to STEP.

| Phase | Status |
| --- | --- |
| 1. Load chain | done |
| 2. Analytical structural layer | done |
| 3. Geometry kernel (CadQuery) | done |
| 4. Browser UI (FastAPI + three.js) | done |
| 5. AI layer (Claude tool-use) | done |
| 6. FEA (scikit-fem + tetgen) | done, except fitting the calibrated factors |
| 7. Block envelope | done |
| 7. Optimisation | done |

## Running it

Requires Python 3.10+.

**On Windows**, Python is usually reachable as `py` rather than `python` -- a
bare `python` is often the Microsoft Store stub, which exits without doing
anything. `run.bat` finds whichever works and tells you what is missing, so
start there:

```
run.bat                        it will name your interpreter and what to install
py -3 -m pip install -r requirements.txt
run.bat                        again, to start the front end
```

The dependency install is not small: cadquery pulls OpenCascade, which is a
few hundred megabytes. Everything except the 3D viewport and the `geometry`
commands works without it.

**On macOS or Linux:**

```
pip install -r requirements.txt

./run.sh                                       # opens the browser front end
python -m psrt margins examples/ls3.json --rpm 6600
python -m psrt margins examples/ls3.json --explain rod
python -m psrt loads   examples/ls3.json --rpm 6000
python -m psrt curve   examples/ls3.json
python -m psrt show    examples/ls3.json -v    # every parameter and its bounds
python -m psrt export  examples/ls3.json --out sweep.csv --rpm 6000
python -m psrt compare before.json after.json
python -m psrt materials -v
python -m psrt new my-engine.json --name "My engine"

python -m psrt geometry examples/ls3.json                  # mass properties
python -m psrt geometry examples/ls3.json --check          # exactness check
python -m psrt geometry examples/ls3.json --export ./cad   # STEP files
python -m psrt geometry examples/ls3.json --profiles       # bore shape study

python -m psrt serve examples/ls3.json                     # the front end
python -m psrt levers examples/ls3.json --objective torque \
       --lock engine.stroke --lock engine.rod_length       # what can I change?
```

`run.sh` / `run.bat` with no arguments starts the front end; with arguments
they pass straight through to the CLI.

On Windows use `run.bat` or `python -m psrt ...`.

`margins --explain <component>` is the one worth knowing about. It prints the
equation used, its source, the operating condition, every Marin factor, every
input, and every caveat. It is also exactly what the AI layer will read in
phase 5 when you ask it why something is limiting.

## What is in here

```
psrt/
  state.py        design state, mutability classes, change log
  schema.py       engine and operating parameters, derivations, migration
  schema_structural.py  component geometry and allowables
  units.py        SI throughout; conversion only at the display edge
  kinematics.py   exact slider-crank, analytic derivatives
  combustion.py   single-zone Wiebe model, or an imported pressure trace
  loads.py        force resolution, torque, friction, brake output
  thermal.py      lumped-conductance temperature map
  sections.py     I-beam, tube and bolt section properties
  fatigue.py      Marin factors, Goodman/Gerber, S-N life, preloaded bolts
  materials.py    material database with temperature derating
  margins.py      the Margin record and the report
  components/     nine structural models, one module each
  geometry/
    properties.py exact volume, centre of mass, inertia tensor
    build.py      parametric piston, pin, rod and sleeve
    profiles.py   circular, polygonal and oval bore cross-sections
  fea/
    mesh.py       surface extraction, welding, tetrahedral meshing
    solve.py      linear elastostatics on constant-strain tets (scikit-fem)
    cases.py      four component load cases with real boundary conditions
    field.py      boundary surface and nodal stress, for the viewport
    calibrate.py  the attempt to fit the phase 2 idealisations, and why it stopped
  optimise.py     constrained search, and every reason to doubt its answer
  envelope.py     what the block allows: every limit, and how much each is worth
  cascade.py      what else moves when the bore moves
  sensitivity.py  ranked levers: what moves an objective, and how far
  server/
    app.py        FastAPI endpoints over the same pure evaluate()
    session.py    the live design, undo, and staged proposals
    static/       the page: three.js viewport, rail, charts, chat
  ai/
    tools.py      the thirteen tools the model drives
    briefing.py   the engineering system prompt
    agent.py      the tool-use conversation loop
  evaluate.py     evaluate(state) -> metrics, the pure function
  cli.py          the interface
tests/            416 tests, including hardware validation against the LS3
examples/         LS3 and a generic 2.0 L I4, with per-parameter provenance
```

## The three ideas the rest is built on

**`evaluate(state) -> metrics` is a pure function.** No hidden state, no side
effects, deterministic. The optimiser, the AI layer, undo and A/B comparison in
later phases all depend on that property. `state.with_changes(...)` returns a
new state, so a change can be evaluated without touching the live design --
which is what makes it safe to let an AI drive in phase 5.

**Every parameter carries a mutability class.** `free`, `bounded`, `locked` or
`derived`. A write to a locked parameter is refused with the reason attached.
The bore's lower bound is the as-built bore, because material comes off a block
and never goes back on; its upper bound comes from bore spacing and minimum
wall. This is not advisory. Loading the LS3 example fails if the block is
described after the bore rather than before it, because the bore's bounds do
not exist yet.

**Every margin carries its provenance.** Not just a safety factor: the
equation, where it comes from, the crank angle and speed it was evaluated at,
every input, and what the model does not capture. A safety factor you cannot
trace is a rumour.

## What phase 2 computes

Nine components, twenty-five margins.

| Component | Checks |
| --- | --- |
| Sleeve | Lame thick-wall hoop stress; wall thickness remaining |
| Crown | Plate bending at edge and centre, each at its own temperature; thermal gradient stress; deflection; high-cycle fatigue; Coffin-Manson thermal fatigue |
| Ring lands | Land bending at the groove root; groove flank pressure; land fatigue |
| Skirt | Specific pressure against scuffing, with the pV factor alongside |
| Gudgeon pin | Beam bending between the bosses; Kolbenschmidt ovalisation; fully reversed bending fatigue |
| Small end | Projected bearing pressure, in compression and in tension |
| Connecting rod | Shank tension; Euler/Johnson buckling about both axes; full-cycle fatigue |
| Rod bolts | Joint separation; thread root fatigue; yielding at proof load |
| Big end | Projected specific pressure; Ocvirk minimum oil film |

Fatigue uses the Marin method throughout: surface, size, load, temperature and
reliability factors, then Goodman mean-stress correction and an S-N life. Every
factor is reported, because a safety factor of 1.8 means one thing when the
surface factor is 0.9 and another when it is 0.26.

Temperature is modelled because it has to be. 2618-T61 keeps 88% of its
room-temperature yield strength at 150 C and 28% at 300 C, and a crown runs
hotter than that. Sizing a crown against room-temperature properties overstates
it roughly threefold.

## What phase 3 adds

Four parametric solids -- piston, pin, rod, sleeve -- each a feature sequence
driven by the design state, so changing a parameter rebuilds real geometry
rather than scaling a mesh. One solid feeds four consumers: exact mass
properties, STEP export, a watertight body for phase 6 to mesh, and a
tessellation for the phase 4 viewport.

**The analytical layer never calls the kernel.** A piston rebuild takes about
400 ms and the phase 7 optimiser needs thousands of evaluations a second, so
the fast path stays closed-form. The kernel runs for display, for meshing, and
for the *exactness check*: rebuild the solids, compare their true mass and
centre of mass against what the design state claims, and report the drift.
`--adopt` writes the measured values back.

Two results worth knowing:

**The rod's centre of mass is now measured.** It sets the reciprocating /
rotating split, and it was previously the familiar "about 0.7 of rod length"
rule. Building the rod properly -- tapered shank, bolt bosses on the big end --
reproduces that 0.700 almost exactly on the generic engine. Leaving either
feature out puts it at 0.62, and that error lands straight in reciprocating
mass, inertia force and rod tension. The rule of thumb is right; the reasons it
is right are not obvious, and two of them are easy to omit.

**The hexagonal bore question has a number.** `--profiles` builds every cross
section inside the same block envelope and compares them. For the LS3 a
hexagonal bore displaces 17.3% *less* than the circular one -- 5147 cc against
6223 cc -- because a circle encloses the most area for a given maximum width,
and the envelope is a maximum width. To gain area the corners would have to
push past it, into material the block does not have. That is before asking how
a piston ring seals a corner.

## What phase 4 adds

A browser front end over the same `evaluate()` the CLI calls. There is no
separate web model of the design: the rail renders `/api/state`, an edit posts
back, and the constraint layer answers.

* **The viewport** builds its meshes from `geometry.tessellate()` and places
  them on real kinematics. The crank slider drives piston travel, rod obliquity
  and pin position straight from the load sweep, so what moves on screen is the
  same motion the forces were computed from. Section view clips on the pin axis
  and opens the piston up; force vectors at the pin scale live with pin load
  and side thrust.
* **The rail** shows every parameter with its bounds, its lock state and the
  reason behind each one. A refused write surfaces that reason rather than
  failing silently. Try to set the LS3's as-built bore and the page tells you
  the block already exists.
* **The charts** are cylinder pressure, pin force with side thrust, and torque
  against crank angle, sharing a cursor with the slider. Colours come from the
  validated data-viz palette; safety factors use the reserved status colours and
  always carry their number, so nothing is encoded by colour alone.
* **Propose / commit / revert** already works end to end, and reports what got
  *worse* alongside what improved. That is phase 5's tool layer arriving early:
  a proposal evaluates against a copy, so the live design is untouched until you
  commit. The AI layer is wiring, not new machinery.

three.js is vendored under `psrt/server/static/vendor`, so the tool draws
without an internet connection.

## What phase 5 adds

An assistant that reasons about the design, in the right-hand rail.

**It has no calculator.** It does not simulate and does not compute. Every
number it states comes from one of thirteen tools that call the same pure
functions the CLI does. That separation is the whole design: it is the
difference between an engineering tool with a conversational interface and a
chatbot that makes up numbers about pistons.

**Refusals are answers.** A write to a locked parameter, or outside a bounded
range, comes back as a structured explanation of the constraint. The model has
to reason around a real limit. A value that looks like millimetres is caught
and named as a unit mistake rather than bounced off the bore's upper bound,
because "you passed 103.5 and this takes metres" is more use than "above
0.10376".

**Nothing commits itself.** `propose` evaluates against a copy and returns the
consequences including what got *worse*; the proposal arrives in the chat as a
card with Accept and Discard, and only you press them.

**`rank_levers` answers "what are my options"** properly. It perturbs every
unlocked parameter, then finds how far each can actually move before a
parameter bound or a safety factor stops it, and says which. A steep lever
with no headroom ranks below a shallow one with plenty -- which is exactly
what a bare derivative hides. Available from the CLI too, as `psrt levers`.

The briefing carries the conventions, the load chain, the two governing crank
angles, what each margin means and which lever moves it, and -- importantly --
which numbers in this tool are calibrated rather than derived, so the model
does not present a fitted constant as physics.

### Turning the assistant on

It needs an Anthropic API key. Get one from
[console.anthropic.com](https://console.anthropic.com/) under API keys, then
pick whichever of these suits you:

**A `.env` file (easiest, and what the tool expects).** Copy the template and
put your key in it:

```
copy .env.example .env        (Windows)
cp .env.example .env          (macOS / Linux)
```

Then open `.env` and replace the placeholder. It is listed in `.gitignore`, so
the key will not be committed. The tool reads it on startup.

**An environment variable**, if you would rather. On Windows, `setx` is
permanent but does NOT affect the terminal you type it in -- open a new one
afterwards:

```
setx ANTHROPIC_API_KEY "sk-ant-..."      Windows, permanent, needs a new terminal
set ANTHROPIC_API_KEY=sk-ant-...         Windows, this terminal only
export ANTHROPIC_API_KEY=sk-ant-...      macOS / Linux, this shell only
```

An environment variable wins over the `.env` file if both are set.

Restart the tool either way. `PSRT_MODEL` overrides the model. Without a key
the Assistant pane says so and everything else works unchanged; the status is
visible at `/api/ai/status`, which reports where the key was found.

## What to distrust

Every input carries a `source`: `published`, `measured` or `estimated`. In the
LS3 example the bore, stroke, rod length, bore spacing and journal size are
published; every mass and every section dimension is an estimate reverse-
engineered from published part weights.

Beyond that, five caveats worth repeating:

0. **The solids are idealisations.** No fillets, ring-groove chamfers, valve
   reliefs, oil drain holes or forging draft. Each of those removes material,
   so a geometry mass runs slightly heavy, and the exactness check says so.
   The LS3's piston and pin land within about 1% of their published weights;
   its rod does not match a commonly quoted 464 g figure, and the geometry is
   believed over it -- that number appears to describe the LS7 titanium rod
   rather than the LS3 powdered-metal one.
1. **Three geometric idealisation parameters were calibrated, not derived,
   and FEA has now shown two of them cannot be fixed by fitting.** See
   "Calibration: what it found". The pin's is uncertain by 76%; the crown's
   rests on a model whose deflected shape is inverted relative to the part.
   `piston.crown_support_radius_fraction`, `pin.support_span_factor` and
   `piston.skirt_bearing_arc` were chosen so that production hardware lands
   just above its limits. The crown stress goes as the square of the first of
   them. Phase 6 FEA is what replaces these with fitted values.
2. **No lubrication model.** Scuffing and bearing film are handled by specific
   pressure proxies. Those are industry practice, not physics.
3. **Thermal-mechanical fatigue is a screening check.** The answer is dominated
   by a single constraint factor, and the plastic strain is a small difference
   between two larger numbers.
4. **Volumetric efficiency and friction are calibration curves** fitted to two
   published dyno points. Neither touches the structural loads.
5. **Wall heat loss is one lumped fraction.** It is a stand-in for a heat
   transfer model.

## Validation

`python -m pytest tests/ -q` runs 416 checks.

**Physics that has to close:**

* Integrated instantaneous torque equals IMEP_net x displacement to better than
  0.001%, which ties the thermodynamics and the mechanics together.
* With fuel energy zeroed, `p V^gamma` is constant across the closed period to
  1e-6 relative standard deviation.
* Analytic derivatives match five-point numerical differentiation, and the
  closed forms at both dead centres.
* Euler and Johnson column curves meet at the transition slenderness without a
  jump.
* The Ocvirk bearing solution round-trips: the eccentricity returned carries
  the load it was given.
* Solid volumes and inertia tensors match their closed forms, and the
  OpenCascade convention is pinned down by translating a box: its
  `MatrixOfInertia` is centroidal, not about the origin. This module had that
  backwards at first, and the test exists so it cannot happen again.
* The lofted rod shank profile encloses exactly the area the closed-form
  section properties assume, so the mass model and the stress model describe
  the same shape.
* Both bundled examples are geometrically self-consistent: every mass in them
  is the mass of the part that would be made.
* The API never emits NaN or Infinity, which JSON cannot carry and which would
  break the front end silently.
* Piston travel out of `/api/sweep` equals the stroke exactly, and the rod
  rotation it sends is continuous through TDC rather than wrapping at 180.
* A proposal leaves the live design's fingerprint unchanged; a refused write
  is a 200 with a structured reason, not a 500.
* The page itself was rendered in a headless browser and checked at four
  window sizes and display scalings: no overflow, 145 parameters in the rail,
  four meshes, ten margin bars, no console errors.
* The whole AI loop is tested without an API key, by driving it with a
  scripted fake model: a tool round-trips and its result reaches the model, a
  proposal never commits itself, a runaway tool loop is capped, a failing tool
  is reported rather than swallowed, and a missing key is a message rather
  than a crash.
* `rank_levers` is checked on the case that once broke it: the probe step was
  larger than the LS3 bore's entire remaining headroom, so both probes were
  refused and the most important lever in the engine vanished from the
  ranking.

**The FEA solver, in `test_fea.py`:**

* The patch test is exact. A uniform strain field imposed on an irregular tet
  mesh is reproduced to machine precision at three refinements -- which is the
  only way to know the element formulation, the assembly and the stress
  recovery are all right at once.
* Uniaxial extension returns Young's modulus and Poisson's ratio back.
* Rigid-body modes are detected by testing the six modes against the restraint
  set rather than by an equilibrium residual, because `K u = 0` for a rigid
  mode makes the residual blind to exactly the failure it is meant to catch.
* The assumptions the restraint code rests on are asserted, not trusted:
  that skfem preserves node order and lays out degrees of freedom node-major.
  `basis.get_dofs(selector)` evaluates its predicate at boundary *facet
  midpoints*, so single-node constraints silently matched nothing until this
  was found.
* Stress recovery is invariant where it must be: a rigid translation produces
  zero stress, a hydrostatic strain produces zero von Mises.
* Bending stress matches beam theory to within 3% on a refined cantilever,
  sampled at mid-span where Saint-Venant says beam theory applies.
* A pressurised sleeve matches the Lame thick-cylinder solution to 1.5% --
  on real CAD geometry rather than a block, so it checks the mesher and the
  solver together -- and the answer moves by less than 2% when the mesh is
  refined 2.6x.
* Constant-strain tetrahedra are stiff in bending, and a test asserts that
  rather than leaving it to be discovered: the cantilever tip reaches 95% of
  the Timoshenko value at 27,000 elements, converging from below and never
  overshooting. Stress is accurate; the deflections are not precise.
* Every part meshes within a factor of 1.5 of the element count asked for,
  and a coarser target really does give a coarser mesh.
* The surface the viewport draws is closed, wound outward, and encloses
  exactly the right volume -- so it cannot render with holes or inside-out
  lighting.
* Welding closes the surfaces. OpenCascade triangulates each face
  independently and never merges the seams -- the gudgeon pin came out with
  148 duplicate vertices and 292 open edges, which is what was segfaulting
  tetgen. All four parts are watertight now.
* Tessellation actually honours its tolerance. OpenCascade caches a
  triangulation on the shape and silently ignores a finer request until the
  cache is cleared, which had been quietly breaking phase 3's `tessellate()`
  since it was written.

**Hardware, in `test_validation_structural.py`:**

* An LS3 survives its own rating. Nothing fails at peak torque or at redline.
* Its margins *cluster*: at peak torque seven independent constraints sit
  between 1.2 and 2.0. Calibrating three parameters cannot drag twenty-five
  margins from nine components into a narrow band unless the underlying models
  are broadly right. That clustering, not the pass, is the evidence.
* The LS3 comes out bore-wall limited, which it famously is, from a purely
  geometric calculation nothing was calibrated against.
* Trends no calibration was aimed at: over-speed moves the binding constraint
  to the rod, a lighter piston relieves it, overboring eats the wall margin,
  and deliberately bad designs -- a paper-thin crown, a spindly rod,
  under-torqued bolts -- are all caught.

One result worth knowing about: **rod fatigue is not monotonic in engine
speed.** It is driven by the swing between compression at peak firing and
tension at overlap TDC. Raising speed grows the tensile half and shrinks the
compressive half, so the alternating stress dips through the middle of the rev
range. The LS3 rod is better off at 8000 rpm than at 4600, and only goes
critical past 10,000. The test suite pins that behaviour down.

## A load that lands nowhere

The worst failure this tool can have is a wrong answer that looks like a good
one. A stress field of all zeros paints a uniformly green part and reads as
"this component is fine".

So before solving anything, the solver now checks what was actually applied,
reports the resultant and the total magnitude in the notes, and refuses three
ways: a load that matched no facets, a load that lands entirely on restrained
nodes and is reacted before it can stress anything, and a load that produces
no displacement anywhere.

Writing that check found two real errors in the load cases, neither of which
had made itself visible:

**The pin and rod were over-loaded.** Both applied a uniform traction over a
curved contact and sized it by PROJECTED area. The arc is longer than its
projection, so the pin delivered 77 kN for a 48.5 kN load -- a factor of
pi/2 -- and the rod 88 kN for 48.6 kN. Stresses were correspondingly high.
Both now use a `Bearing`, which is rescaled to its resultant after assembly
and so cannot be got wrong by the mesh or the curvature. The pin's ratio fell
from 1.33 to 0.88 and the rod's from 2.04 to 1.23.

**The rod was loaded on the wrong wall.** Under compression the pin drives
the rod at the crank, so it bears on the side of the small-end bore facing
the big end. The case had it on the near side. A traction vector ignores
surface normals, so it applied happily to the wrong wall and produced a
plausible-looking answer; a bearing follows the normal and refused outright,
which is how this was found.

The two cases that were already applying a true pressure -- the crown and the
sleeve -- did not move at all, which is the check on the check. And the
sleeve still lands within 3% of Lame.

The first version of the guard tested the load RESULTANT, and promptly
rejected the sleeve: a pressurised cylinder carries a large load and a
resultant of exactly zero, by symmetry. It tests the total magnitude now.
There is a test for that specific mistake.

## When the mesh is the problem

A stiffness matrix can be singular with the part perfectly restrained, and
the symptom points at the wrong thing. The solve returns non-finite
displacements; NaN then fails every comparison in the code that follows, so
the residual never exceeds its tolerance and the stress recovery zeroes the
non-finite tensors by design. What reaches the screen is a part at 0 MPa
everywhere, painted uniformly green -- a wrong answer wearing the face of a
healthy one. The first error message this produced blamed the restraints,
which held 666 nodes and were entirely innocent.

Several mesh faults do it, and `tet_mesh` repairs all of them before the mesh
is ever handed to a solver:

**Coincident nodes** -- two points at the same coordinates leave a crack
through the part. The elements either side share no index, so nothing carries
load across, and each piece moves independently. Welded first, because every
repair after this one depends on the connectivity being real.

**Flat elements** -- a tetrahedron with four coplanar corners has no volume,
contributes nothing, and its own element matrix is singular. Dropped.

**Inverted elements** -- a negatively-oriented tetrahedron contributes
NEGATIVE stiffness, which destroys the positive-definiteness the solver
relies on. Swapping two of its nodes turns it the right way without moving
anything, and different tetgen builds order nodes differently, so this is
clean on one platform and not on another.

**Orphan nodes** -- points tetgen returns that no tetrahedron references.
They contribute nothing to the matrix, so their rows and columns are exactly
zero, and the system is singular however well the part is held. Dropped, and
the elements reindexed.

**Disconnected islands** -- elements sharing no node with the main body, each
floating free with its own six rigid-body modes that restraining the part
does not touch. Debris under 1% of the mesh is discarded and reported;
anything larger is refused outright, because at that size it is more likely a
real piece of the geometry than an artefact, and quietly deleting part of the
component would be worse than failing.

**Pinched chunks** -- and this is the one that survived every check above.
The island test joins any two elements sharing a single NODE, so a chunk
hinged on one vertex, or on one edge, counts as connected. It is not: it
rotates about its hinge at zero strain and therefore at zero energy, so the
matrix has a null space no restraint on the part can reach, and the island
counter reports nothing wrong. A real solid mesh is FACE-connected -- you can
walk from any tetrahedron to any other through shared triangles -- so the
graph is built on shared faces instead, and anything outside the main
component is attached by an edge at most. Same 1% rule: debris goes, a real
piece refuses. Dropping a pinched chunk can pinch off what it was holding, so
the pass repeats until it is clean.

None of these reproduce on every platform. These meshes come out clean from the
Linux build of tetgen and singular from the Windows one, which is exactly why
it is handled rather than hoped about. The tests build broken meshes by hand
instead of waiting for a mesher to produce one.

### The rod, and why a correct solve depended on the machine

The rod was the last of these and the least like the others. Its mesh was
clean -- zero orphans, zero islands, zero pinches, one face-connected
component, every solid valid per OpenCascade -- and it returned NaN on
Windows and the right answer on Linux from the same source.

The cause was conditioning, and the rod earns it honestly. It is the one part
whose mesh spans a huge range of element sizes: the I-beam web meets the
small-end boss at an acute angle, tetgen fills that corner with slivers, and
the smallest element comes out at **1.7e-15 m3 against a median of 3.1e-9**,
a ratio of 1.8 million. The pin and the sleeve run 37:1. Element stiffness
scales with element size, so the matrix inherits the spread: the rod's
diagonal runs over four orders of magnitude, against two on the piston and
one on the pin. At that spread the condition number is high enough that
whether a factorisation survives comes down to the pivoting choices of
whichever SuperLU the local scipy was built against. One machine, one part.

So the solver rescales before it solves. Symmetric diagonal equilibration --
the substitution u = D^-1/2 y with D the diagonal of K -- gives an equivalent
system with a unit diagonal. It is exact, not an approximation: the solution
transforms straight back, and a test asserts the two solves agree to nine
figures. It costs one sparse product. On the rod it takes the condition
number to about 1.5e6, which any factorisation on any platform handles
without noticing.

Then the answer is CHECKED rather than assumed. The relative equilibrium
residual of the condensed system is a few flops, and it catches the failure
mode `isfinite` cannot: a solve that comes back finite and wrong. Only if the
check fails does the solver climb a ladder -- plain direct, then conjugate
gradients, then MINRES -- and which rung was used is reported in the result
notes, because it is a statement about the mesh and not an implementation
detail. MINRES is the interesting rung: it tolerates a singular matrix, and
for elasticity that is worth having, because a stiffness matrix's null space
is made of rigid-body motions and those carry no strain. A free-floating
chunk leaves the displacements defined only up to that motion and **the
stresses still exactly right**.

`python examples/diagnose_fea.py` prints all of it for all four parts --
versions, BLAS, repair counters, face components, element-size spread, and
which rung solved each one.

Four guards now stand between a bad solve and a green part: a load that
matched no facets, a load reacted entirely by restraints, non-finite
displacements, and a finite solve that recovers zero stress. A fifth checks
the surface field the viewport actually paints, because the solver guards its
own answer and not what is drawn from it.

`/api/version` reports which of these the RUNNING process has, read off the
loaded code objects rather than the files on disk. The first version checked
the source on disk and answered the wrong question: the disk is current the
moment an edit lands, while the process keeps whatever it imported at
startup.

## Calibration: what it found

Fitting the phase 2 idealisations from FEA was the point of building an FEA
layer at all. The attempt is in `psrt/fea/calibrate.py`. **It did not produce
new constants, and the reasons are worth more than the numbers would have
been.** Nothing in the design state was changed; a test asserts that.

One thing did work, and everything below rests on it. Loads and REACTIONS are
now both applied as cosine-distributed contact pressures
(`psrt.fea.solve.Bearing`), scaled to deliver an exact resultant. The load set
then sums to zero, the restraint only has to remove rigid-body motion, and
the singularity it used to invent is gone -- the pin's 99.5th-percentile
stress fell from 733 MPa to 456 MPa, almost all of the difference being
artefact. Comparisons are also made on an integral, the bending moment
obtained by integrating axial stress over cross-sections, rather than on a
peak. A moment is insensitive to local concentrations in a way a peak is not.

### The pin span cannot be fitted this way

`pin.support_span_factor` says where the boss reaction acts. Solve, extract
the moment, invert for the span, done -- except that a linear analysis takes
the contact pressure distribution as an **input**, and the resultant of a
prescribed distribution sits at its own centroid. So the fitted span comes
back as wherever the pressure was put:

| reaction spread over | effective span | fitted factor |
| --- | --- | --- |
| inner third of the boss | 30.3 mm | 0.32 |
| inner two thirds | 38.0 mm | 0.71 |
| the whole boss width | 44.5 mm | 1.04 |

The method reproduces its own assumption to three figures. Settling it needs
a contact solution -- the pin bends away from the boss outboard of its inner
edge, so the contact patch is part of the answer and shrinks under load. That
is a Signorini problem, not a linear solve.

So `pin_span_bracket` returns a bracket instead: the factor is somewhere in
**[0.32, 1.04]**, which is a **76% uncertainty in the pin's bending stress**.
The phase 2 guess of 0.50 sits inside it. That is the honest statement, and
it is more useful than a false constant.

### The crown model has the wrong shape

`piston.crown_support_radius_fraction` idealises the crown as a clamped
circular plate. Fitting its radius from the centre moment -- which avoids the
unfilleted crown-to-wall corner, a genuine singularity -- gave a value that
varied **87%** across five geometries, and systematically with crown
thickness: 0.165 at 5.2 mm, 0.243 at 6.5 mm, 0.369 at 8.8 mm.

A power law `0.236 x (t/R)^1.53` fits all five points to within 3%, including
two held out. But it implies crown stress **rising** with thickness, which is
impossible for a pressure-loaded plate -- so the fit is curve-fitting a model
of the wrong shape, and extrapolating it would be dangerous.

Looking at the deflected shape says why. Under gas load the crown centre
moves **up 12 um** while the rim moves **down 53 um** -- the exact inverse of
a plate. The pin boss pad, which the geometry builder carries up to the crown
underside, is over half the bore wide, so the crown centre is *supported*
rather than spanning. `crown_deflection_profile` measures this and a test
asserts it.

That makes three places where two models in this tool disagree about the same
part, all found by measurement rather than by reading code: the rod I-section
(fixed), `piston.skirt_length` (fixed), and now the crown. The constants stay
as they are -- labelled calibrated guesses, which is at least honest about
what they are -- until the crown is modelled as what it is: a disc supported
on a wide central pad, not a clamped plate.

## Reading a stress field

Press a **Stress** button and the part is solved and painted. Two things
about that picture are worth knowing.

**It covers the whole crank cycle, from one solve.** Drag the crank slider
and the field follows. That is not three hundred solves: the analysis is
linear and each case has a single load pattern, so the stress everywhere is
exactly proportional to the load, and the field at any angle is the solved
field times `load(theta) / load(reference)`. A test asserts that
proportionality holds to machine precision, because the whole feature rests
on it.

The one real assumption: when the load REVERSES -- the pin and rod go into
tension around overlap TDC -- the part bears on the opposite side of its
contact, so the true boundary conditions are mirrored rather than negated.
The magnitude is right and the distribution is mirrored from what was solved.
For a tensile half-cycle an order of magnitude lighter than firing that is a
reasonable place to stop; for a case where tension governed it would not be.

**The colours answer one of two different questions.** The mode selector on
the legend switches between them:

| mode | scaled to | answers |
| --- | --- | --- |
| distribution | this field's own 99th percentile | *where* the load concentrates |
| vs yield | the material's allowable at temperature | *how close* it is to letting go |

The distribution view is the one to reach for when reshaping a part, and it
is also the one that misleads: a red patch means "highest here", not "about
to yield". The yield view fixes the scale to what the material can actually
carry, so on a healthy design most of the part sits at the cold end of the
ramp.

That allowable is taken at the temperature the part actually runs at, not
from the room-temperature catalogue. It matters most where it is easiest to
get wrong: the LS3 crown runs at 232 C and A390 has lost half its yield
strength by then -- 158 MPa, not 310. The derating tables are keyed in
kelvin while the thermal map reports Celsius, and mixing the two silently
returns the cold number; there is a test for that specific mistake. Grey iron
has no yield point at all, so the liner is referenced to ultimate strength
and the legend says so.

**The peak is usually not a stress.** The single hottest element normally
sits directly under an applied restraint, where a rigid boundary condition
invents a singularity that refines toward infinity. The caption says so every
time. Read the body of the part.

**A field belongs to the design it was solved on.** Change any parameter and
it is dropped, with a message saying why. A stale stress picture is worse
than no picture, because it looks exactly like a current one.

## Phase 7: the optimiser

    python -m psrt.cli optimise examples/ls3.json torque
    python -m psrt.cli frontier examples/ls3.json torque

or the **Optimise** tab in the browser.

The optimiser was built last on purpose. It is only as trustworthy as the
model underneath it, and an optimiser is the best tool ever devised for
finding a model's weak spots and driving straight into them. So it is built
to be suspicious of its own answer.

**It cannot cheat.** Every candidate is written through `DesignState.set`, so
a parameter bound -- including the block envelope's overbore ceiling --
refuses the search exactly as it refuses a person. The returned design is
re-validated from scratch rather than trusted because the search said so.

**It re-measures what it optimised.** The search runs on the fast analytical
layer, whose masses are stored numbers. Afterwards the solids are rebuilt and
the design re-evaluated; if the verified answer differs from the one the
search believed, the result says so instead of reporting the optimistic one.

**It says what stopped it, and what that is worth.** An optimum is a
statement about which constraint is binding, and if that constraint's model
is one of the phase 2 calibrated guesses rather than an FEA-fitted value, the
result says so -- an optimum sitting on a guess is worth what the guess is.
It also flags levers pinned against a bound (the answer is set by the bound,
not the physics) and levers that moved the objective *not at all*, which in
this model usually means a gap between two models rather than real
insensitivity.

### Levers are grouped by what you would have to do

Turned loose on everything, the search answers "how do I get more torque?"
with a longer stroke and a 55% longer burn. Both are true, and both are a
different engine plus a remap rather than a redesigned piston system. Worse,
burn duration is the easiest way to exploit the edges of a single-zone Wiebe
model, so the biggest number in the answer would be the least trustworthy
part of it.

The default is `machining, piston, rod` -- this block, with remachined parts.
`rotating` (a new crank) and `tuning` (a calibration change) are opt-in and
say what they cost.

### The frontier is the honest answer

"What is the optimum" has no answer without knowing what you are willing to
spend. `frontier` prices it:

| min safety factor | brake torque | binding |
| --- | --- | --- |
| 1.243 (as it is) | 572.3 N m | pin: ovalisation |
| 1.342 | 560.1 N m (-2.1%) | crown: high-cycle fatigue |
| 1.541 | 533.5 N m (-6.8%) | crown: high-cycle fatigue |
| 1.740 | 508.4 N m (-11.2%) | crown: high-cycle fatigue |

Two things worth reading off that. The LS3 sits *on* its own worst margin, so
the first row is what it already does. And the binding constraint moves from
the pin to the crown immediately above the current design -- which is why
"strengthen the pin" stops paying almost at once.

Epsilon-constraint rather than a population method: slower, but every point
is a real design satisfying every constraint, arrived at deterministically. A
front whose points are not individually buildable is a picture, not an answer.

### One thing the optimiser found

Asked to minimise reciprocating mass, it refuses and explains why: the design
already sits on its safety floor, so every improving direction costs margin.
Given room (`--min-sf 1.20`) it finds 34 g -- about 4.2% -- for 0.03 of safety
factor.

On the way it reported that `piston.skirt_length` moves no mass at all, which
turned out to be a real gap: the structural model read it for skirt bearing
area and the geometry builder never read it, so the two had drifted 11% apart
on the LS3 and 20% on the default.

They are still separate quantities -- a real skirt is barrelled and relieved,
so the CONTACT length is legitimately shorter than the panel -- but contact
longer than the panel is not legitimate, and `refresh_bounds` now caps it.
The LS3's dropped from 30.0 to 26.6 mm, which cost the skirt margin 1.72 ->
1.53.

That had a consequence worth recording. At 9,000 rpm skirt scuffing and rod
fatigue now finish within 6% of each other, so a test that used to assert
"overspeed moves the limit to the rod" was asserting noise -- 6% is well
inside the uncertainty of a skirt model this README already calls a proxy
rather than physics. It now asserts what the model genuinely supports: that
both inertia-driven limits overtake the gas-driven ones.

## Phase 7: the manufacturing envelope

Every other part of this design can be drawn freely. The block cannot -- it is
an existing object, and machining is subtractive. So the bore has a hard floor
at its as-built size and a ceiling set by whichever internal feature the wall
reaches first.

    python -m psrt.cli envelope examples/ls3.json
    python -m psrt.cli overbore examples/ls3.json 103.5

The ceiling is written onto `engine.bore` as a bound, not checked at the point
of use. That matters: every write path in the tool -- the CLI, the API, the
assistant -- goes through `DesignState.set`, so an overbore past the block's
limit is **refused with its physical reason** rather than computed and
regretted afterwards.

### Not every number is worth the same

The inputs to this are not equally knowable, and presenting them as if they
were would be the tool lying by layout. Each limit carries the tier of its
weakest input:

| tier | what it covers | where it comes from |
| --- | --- | --- |
| 1 | bore, stroke, deck height, liner type | manufacturer specs, or measured on the block |
| 2 | the largest bore a piston is sold for | aftermarket catalogues |
| 3 | coolant jacket, oil gallery positions, the real wall at the thin spot | nobody publishes these |

Tier 2 is the one worth building around. If a piston maker has tooled up for a
+0.010 in. piston, the industry has already established that overbore survives
in the field -- and that is better evidence than anything this tool could
derive from first principles. On the LS3 it is the binding limit, at 103.505 mm
(4.075 in., which Wiseco, Mahle, Manley and JE all catalogue), and it is both
**tighter and more trustworthy** than the tier 3 bore-to-bore figure of
103.76 mm computed from an assumed wall thickness.

A limit whose inputs are missing is reported as unknown, never skipped. A
skipped limit silently raises the ceiling, and the limit nobody could evaluate
is exactly the one likely to bite. If nothing at all can be evaluated the bore
is pinned where it is rather than left free.

### Enlarging a bore is never one change

    python -m psrt.cli overbore examples/ls3.json 103.5

reports displacement, compression ratio, piston mass, inertia, side thrust and
every structural margin -- what got **worse** as prominently as what got
better. Any tool that answers a bigger bore with only the torque gain is lying
to you by omission.

Building that surfaced a real gap: `masses.piston` is a stored number, not an
expression, so the fast layer was blind to the one parameter that most
invalidates it. A bigger bore reported more torque and an *unchanged* piston,
hiding the inertia penalty that makes overboring a trade at all. The cascade
now rebuilds the solids and re-measures both sides.

### Resleeving

Boring oversize and pressing in a thicker ductile-iron sleeve adds material
back and resets the envelope. It is a mode you switch on deliberately, never
something an optimiser discovers on its own.

The first version of this model was wrong in an instructive way: it took the
unsleeved ceiling and subtracted two sleeve walls, which made resleeving come
out *smaller* than the stock block -- and if that were true nobody would pay
for it. What was missing is that interlocking sleeves become the structure
between the bores, so the coolant-wall rule is replaced by a sleeve-to-sleeve
gap near zero. With that, an LS3 comes out at 105.26 mm (4.144 in.), which is
the range real sleeved LS blocks are built to. A test asserts resleeving can
never make a block worse.

## Phase 6: FEA

A linear elastostatic solver on constant-strain tetrahedra (`psrt/fea/`),
built on scikit-fem and tetgen rather than the planned gmsh and CalculiX --
both of those wanted system installs the project would then depend on. All
four parts mesh in under a second, the solver is checked against three closed
forms, and the von Mises field is painted onto the part in the viewport.

    python -m psrt.cli fea examples/ls3.json pin
    python -m psrt.cli fea examples/ls3.json piston
    python -m psrt.cli fea examples/ls3.json rod
    python -m psrt.cli fea examples/ls3.json sleeve

or press one of the **Stress** buttons in the browser front end, which are
built from whatever `cases.py` defines -- adding a case puts a button there
with no front-end change.

The FEA libraries are the only dependencies added after the tool was first
installed, so if the buttons report them missing:

    python -m pip install scikit-fem tetgen scipy

**The negative result worth more than the feature.** This module spent a
whole phase blocked on the gudgeon pin meshing to 390,000 elements against a
25,000 target, whatever it was asked for. The diagnosis at the time was that
CadQuery hands over slivers -- the pin arrives as triangles 1.4 mm around the
circumference and 63.5 mm along the axis -- and that tetgen's quality
criterion could not be reined in.

That diagnosis was half right and the conclusion was wrong. The surface
refinement *this code was doing about it* was the cause. Splitting an
over-long edge at its midpoint never touches the short edge opposite, so the
median aspect ratio **climbs** under refinement, 16 to 28, and tetgen -- which
preserves the boundary it is handed -- had no choice but to flood the volume
to conform to it. Deleting the refinement fixed every part at once. A full
isotropic remesher was written before this was understood; it is not in the
tree. There is a test whose whole job is to stop the refinement coming back.

There are four load cases -- pin bending, crown pressure, rod compression and
bore pressure -- and each one samples the FEA **where its analytical formula
applies** rather than taking a whole-part peak. That distinction is not
cosmetic: the rod's highest stress is in the small-end ring while
`F / A_shank` describes the shank, so the unsampled ratio came out at 5.6 and
meant nothing.

| case | compared in | FEA / analytical |
| --- | --- | --- |
| pin | tension fibre over the loaded span | 0.88 |
| sleeve | bore wall at mid-height | 1.03 |
| rod | shank, between the two end rings | 1.23 |
| piston | crown disc inside the support radius | 1.49 |

Two of those numbers moved a long way once the load was checked rather than
assumed -- see "A load that lands nowhere" below. The pin and rod cases were
applying more force than they were asked to.

The sleeve row is the one to read first. Its analytical model is Lame, which
is exact for that geometry, so its ratio has a known right answer of 1.0 --
and it comes back at 1.03. That is the end-to-end check that the mesher, the
pressure load and the solver agree with each other.

**Still open: the calibration.** Those ratios are not stress concentration
factors and must not be written into the fast layer as if they were. Each one
carries the contact idealisation its case assumed -- that a boss reacts
uniformly across its width, that a crankpin bears evenly over half a bore.
Rigid restraints also invent singularities that refine toward infinity, which
is why the peak column is reported separately and why every ratio is tested
for mesh independence: the pin's, sampled right up to its restraint edge,
read 2.01, 2.39 and 2.38 at three densities before the sample was pulled
clear, and a number that moves with mesh density is not a property of the
part.

Doing the calibration properly means extracting the bending moment
distribution along the pin and fitting the support span to that, and giving
the load and reaction patches a compliant footprint instead of a rigid one.
Until then `crown_support_radius_fraction`, `pin.support_span_factor` and the
notch factors remain calibrated guesses, and caveat 1 above still stands.
