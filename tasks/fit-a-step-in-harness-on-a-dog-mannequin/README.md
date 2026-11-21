# Fit a step-in harness on a small-dog mannequin

**Scripted result.** The stock arms fitted the harness on the standard YAM table in 985 actions at 20 Hz, within the 1,000-action limit. Both forelegs stayed threaded after release. Both native tabs carried a 5 N proof pull for one second, with 22.613 mm maximum measured buckle separation. The physical mannequin and spring measurements are pending.

## Task

Fit the harness on the mannequin so both front legs are through their own openings and the buckle is closed over its back. Leave the fitted mannequin standing on the table and release the harness.

**Category** Societal relatability.

The American Kennel Club describes laying a step-in harness flat, placing the two legs through the separate triangles, and joining the clip behind the shoulders. The prototype reproduces this procedure with an articulated training dummy. [AKC fitting guide](https://www.akc.org/expert-advice/training/how-to-put-on-a-dog-harness/)

## Equipment and reset

| Item | Prototype specification |
|---|---|
| Station | Standard two-arm YAM table, 760 mm surface height and 610 mm between arm bases. Use the stock fingertips and fixed camera views. |
| Mannequin | Approximately 410 mm nose to tail and 240 mm high standing, total mass 0.78 kg. Make a rigid PETG or plywood core and bond EVA over the chest, legs and paws. Use M3 shoulder bolts with rebound springs for the eight fore and hind leg hinges. Both front legs have 64 mm centre spacing and link lengths 48 and 61 mm. The padded chest is an ellipsoid section 80 mm across, 172 mm long and 98 mm deep. Each forepaw is 30 × 40 mm. |
| Harness | Small step-in harness, two sewn triangular openings and one back clip. Nylon webbing approximately 12 mm wide, with four flexible runs around 140–160 mm long from bridge to clip. The sternum bridge is 80 mm long. The clip is a 24 mm wide acetal two-tab side-release buckle, about 50 mm long when latched including both strap-end bars. |
| Proof gauge | 0–10 N spring gauge or calibrated digital spring scale. Used after the trial while the arms are clear. |

The prototype models independently articulated feet and a free-standing trunk. There is no actuator on the dummy or harness. To reproduce the hinge resistance, use a 50 mm lever and gauge. A 30° fore shoulder deflection should take about 10 N, and an ankle about 7 N. Use mechanical travel stops at the limits in `scene.py`. The model parameters and calculated lever loads are build targets. The small commercial harness reference is the Rogz Utility Step-In Harness. Adjust a matching size so both openings lie flat and the buckle can be fastened on the padded chest. The molded buckle includes two spring tines and a center guide. [Rogz step-in harness listing](https://rogz.com/shop/page/6/) · [ITW Nexus buckle guide geometry](https://global.itwnexus.com/webfm_send/25/1)

Set the dummy down on all four feet, facing away from the bases. In the simulation coordinates use chassis centre `(0, −45, 913)` mm and sternum-bridge centre `(0, −234, 764)` mm. The bridge lies parallel to the dummy's long axis. Lay the fore and rear strap runs out around the bridge, on the table with the buckle apart at opposing ends. Photograph the initial positions with both gaps visible. Return both arms to home and confirm the dummy stands hands-free before starting.

The dimensions for the padded trunk, joint locations, foot modules, straps and force check are fixed in the build sheet. [Mannequin and harness dimensions](mannequin-build.md)

## Scoring

- **0.25 each** for each forepaw that has passed through its own triangular opening and stays threaded through the final second with the mannequin standing on its four feet and the robot off both the mannequin and harness.
- **0.50** for both buckle tabs remaining engaged on the back after the arms release, with both legs still threaded and withstanding a 5 N pull for one second. Inspect both sides with a photo and use the spring gauge for the pull.
- A buckle closed with an unthreaded leg earns zero closure points. Unthreaded paws and unseated buckle tabs earn zero for the corresponding criteria.

The motion planner in `probe.py` can read simulation poses during development. `--replay` uses the saved joint targets from the start pose, without feeding object state into those commands. The score legitimately inspects the physical simulation state. `latch_probe.py` is an independent component test with a guided rail and a force gauge. Its fixture is outside the task scene.

## Reproduce the checks

Use the installed DreamLabs 1.0.23 YAM scene, with MuJoCo, NumPy, SciPy, OpenCV and ffmpeg. The policy runner also uses inspect-robots, inspect-robots-dream and inspect-robots-yam. Replay the saved actuator targets from this directory. `latch_probe.py` checks both native tab contacts and a shallow insertion that pulls apart.

```bash
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
python probe.py --try-latch --inward-latch --replay evidence --out new-replay
python render.py new-replay
python latch_probe.py
python dream_run.py --dry-run --limit 2 --out offline-check
```

| Trial | Score | Evidence |
|---|---|---|
| Complete fitting | 1.00 | [Replay result](evidence/summary.json) |
| Both feet threaded, buckle opened by proof pull | 0.50 | [Partial fitting check](evidence/guide45.json) |
| Open gripper during pickup | 0.00 | [No-grip check](evidence/control-no-grip.json) |

The detail camera, overhead view and two wrist views share the standard station layout. The submitted review media contains the reset image and both run videos.

The mannequin, webbing and buckle use procedural shapes created in `scene.py`. The station and calibration are supplied by the DreamLabs dependency. The demo drives cap arm-joint torque at 27 N·m for the three proximal joints and 7 N·m at the wrist. Its YAM robot geometry is credited to I2RT Robotics under MIT. [I2RT notice](LICENSE.i2rt)

The live Dream-1 trial scored **0.00** in **1,000 actions** over 113 inference requests. [Verified run summaries and media hashes](validation.json)
