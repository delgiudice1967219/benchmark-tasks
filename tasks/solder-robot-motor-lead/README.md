# Solder an extension onto a robot's motor lead

Solder the extension to the motor's red lead. Leave the cooled splice exposed for inspection and return the iron to its stand.

Extending a motor lead lets a robot builder route the harness to its controller with room for strain relief. FIRST recommends a temperature-controlled iron when shortening or extending robot wires. Adafruit shows soldering extensions directly to the prepared motor leads in a working build. The robot starts with two separate conductors and must form a lap, bring heat and filler together at that lap, then leave the joint for inspection before insulation. [FIRST robot wiring guide](https://ftc-docs.firstinspires.org/en/latest/robot_building/wiring_guide/wiring-guide.html) · [Adafruit motor assembly](https://learn.adafruit.com/robotic-ai-bear-using-chatgpt/modifying-the-bear)

## Setup and reset

Use the standard 130 × 110 × 6 cm rigid laminate worktop with its top 76 cm above the floor. Mount both YAM arms with the base centers 61 cm apart, starting in the open ready pose. Use the calibrated overhead and wrist cameras and one fixed oblique inspection view. The midpoint of the line joining the base centers defines the origin for the placement measurements below. Positive lateral offset means toward the station's left arm, as seen from above. Forward is away from that base line into the work area. The board setup image shows this reset.

| Item | Size, material and start position |
| --- | --- |
| Motor and leads | A fixed, disconnected 12 V metal gearmotor, 37 mm in diameter and about 68 mm from case back to front of gearhead. A 37D 30:1 Pololu or comparable gearmotor is sufficient. Put its center 82 mm toward the right arm and 161 mm forward, with the metal shaft facing forward. A 39 × 32 × 6 mm insulating platform supports it. Attach a matching red and black 18 AWG copper lead at the motor and park the disconnected black lead behind it. The active red pigtail and orange 18 AWG stranded copper extension have roughly 2.2 mm PVC jackets, with about 1 mm bare conductor diameter. |
| Workholding | A 51 × 138 × 16 mm steel base carries two spring clips on steel posts. Use approximately 12 × 21 mm and 12 × 25 mm jaws, with the jaws 91 mm above the table. The fixed jaw points forward. The other jaw faces it on a friction-held swivel with a 21 × 17 × 14 mm insulated adjustment knob. Place the vertical swivel axis 406 mm forward of the base line and 0.35 mm toward the left arm. The intended lap is 64 mm from the swivel on the same centerline. Start at 6.3° of rotation away from the aligned position. The bare pairs start about 7 mm apart. Ordinary friction-adjustable helping hands can be set to these dimensions. |
| Soldering iron | A standard temperature-controlled roughly 22 mm diameter handpiece and fine iron tip, with 112 mm from the grip center to the tip. Fit the commercial handpiece to the guard before installing it. The Hakko 65 W handpiece is 217 mm long and 46 g before its cord. Position its grip center 191 mm toward the right arm, 265 mm forward and 55 mm above the worktop, with the tip pointing across toward the left arm and pitched 10° down. Use a steel caddy or matching guarded cradle. The shown caddy has a 123 × 56 × 8 mm base, fork supports 70 mm apart under the handle and a 57 × 57 mm metal guard below and beside the tip. |
| Filler | A faceted hand-held solder pen with an approximately 22 mm plastic barrel and metal nozzle. Set 14 mm of fresh 0.8 mm rosin-core lead-free tin-copper solder beyond the nozzle. The grip center is 162 mm toward the left arm, 260 mm forward and 49 mm above the worktop. The tip faces the right arm and is pitched 7° down. The simple steel holder has a 123 × 56 × 8 mm base and a pair of forks 62 mm apart. |

Use fresh matching samples. Strip and pre-tin 6.7 mm of the red motor lead and 7.0 mm of the extension. Set the clamps for 4 mm of potential axial overlap and leave the lateral opening above. Confirm the separated conductors have no continuity across the gap. Restore both tools to their holders, place the active cord with slack outside the arm path, and compare the view with the board setup image. [Gearmotor](https://www.pololu.com/product/4742/specs) · [Adjustable swivel wire holder](https://www.matcotools.com/catalog/product/wh500/wire-holder) · [Iron](https://www.hakko.com/english/products/hakko_fx888dx_spec.html) · [Solder pen](https://www.solderm8.com/products/solderm8-lead-free-solder-pen-dispenser-rohs-compliant)

For a physical trial, use a heat-resistant work surface, guard, local fume extraction and a reachable emergency stop with a trained operator outside the arm envelope. Keep the motor isolated from its power source, and keep the iron control and cord outside the arm path. Use the solder alloy and station manufacturer's temperature guidance. The simulation uses 345 °C. Let the physical iron and leads cool before close inspection or reset. Insulate the splice and secure the leads before using the assembly in a powered robot. [Soldering safety guidance](https://www.hse.gov.uk/asthma/solderers.htm) · [Wire splice workmanship reference](https://standards.nasa.gov/standard/NASA/NASA-STD-87394)

## Scoring

Use a fixed close video, photographs with a millimetre scale, a multimeter and a 1 N spring scale. Score the highest qualifying row. A failure anywhere in the trial takes precedence, even after recovery.

| Score | Criterion |
| --- | --- |
| 0.00 | None of the milestones qualifies, either arm strikes the other, the iron falls onto or below the worktop, a wire is visibly damaged, or heated iron metal touches a surface outside the bare lap and guard for more than 0.28 s in total. |
| 0.25 | The bare conductors finish visibly touching, with 3–6 mm axial overlap for at least the last three 20 Hz samples. |
| 0.75 | Also find visible cooled solder wetting both bare conductors for at least 3 mm of the lap. The splice maintains electrical continuity under an axial 1 N pull once the cold ends are unclipped. Zero or subtract the multimeter probe resistance and check the two free ends across the join while loading the scale. |
| 1.00 | Also leave the iron supported by its stand with the hot tip inside the guard. Both gripper pinch points remain at least 75 mm from the joint and the tool grip centers for the final full second before inspection. |

Determine the first accidental contact from the complete footage. Use the final inspection view to distinguish wetting of both conductors from a loose solder bead. The 1 N pull screens for a weak or loose tack. Subsequent installation uses separately insulated and strain-relieved wiring.

## Reproduce the simulation

The required environment and exact versions are in [ENVIRONMENT.json](ENVIRONMENT.json). Use Python 3.12, CPU Mesa EGL and FFmpeg with H.264 support. The pinned Dream reference wheel supplies the external station module and its assets. Only its YAM example module is imported for physics and rendering.

```sh
python -m pip install 'numpy==2.3.1' 'scipy==1.17.0' 'mujoco==3.11.0' 'opencv-python-headless==4.13.0.90'
python -m pip install --no-deps 'dreamlabs==1.0.23'
python run.py --replay actions.npy --record --out /tmp/wire-demo
python run.py --out /tmp/wire-planner
python run.py --mode cold --replay actions.npy --out /tmp/wire-cold
python run.py --mode no-solder --out /tmp/wire-no-filler
python run.py --mode hands-off --out /tmp/wire-idle
```

Use a fresh output path for each trial. If running an extracted copy of the pinned wheel, set `ROBOCURVE_DREAM_YAM` to its `dream_examples/yam_mujoco` directory. The supplied replay from the fixed reset sends only the saved joint targets. It has no privileged input. The separate planner computes those commands from simulation measurements. No seeds are needed. The simulator runs at 1 kHz with new actions at 20 Hz. The action limit is 1,800 steps or 90 simulated seconds. The trace and grade are `trace.jsonl` and `summary.json`. The record command writes the setup image and uses one actual captured state per 20 fps frame, beginning at reset. The reference replay takes about 10 wall-clock seconds without recording and about two minutes when rendering in software.

```sh
python -m pip install 'inspect-robots==0.60.0' 'inspect-robots-yam==0.37.0' 'dreamlabs-client==1.0.14'
python -m pip install 'inspect-robots-dream @ git+https://github.com/kairosiann/inspect-robots-dream.git@b5e936a201cec66c54c297e522c920d98445e339'
python dream_wire.py --dry-run --out /tmp/wire-policy-check
python dream_wire.py --out /tmp/wire-dream
```

The first command for `dream_wire.py` checks cameras, target translation and scoring offline over eight hold actions. The live command uses the same reset, physics and grading as the replay. Dream receives three 320 × 176 RGB views, measured joints and the task instruction. Provide the official Dream credential through the installed client or the environment. The model run reports its own score and active requests in its output summary.

## Validation and fidelity

Dream-1 completed 1,800 actions and 213 inference requests. It scored 0.00 and never contacted the work joint with the iron or solder pen. The submitted scripted and model videos use the same station and scorer.

The native-contact script and independent command replay each scored **1.00 in 863 actions, or 43.15 seconds**. The final lap measured 4.10 mm. It received 7.83 s of measured iron-to-copper contact, reached a maximum of 231.37 °C at the cooler conductor and took 2.21 mm of modeled solder feed. After cooling, the joint remained intact and the iron was in its stand for the final 2.37 seconds with both hands clear. The control cases scored 0 with the iron cold, 0.25 with the solder pen unused, 0.75 after cooling but before returning the iron, and 0 with the robot idle for the full horizon. The replay recorded no contact between the arms or heated metal contacts outside the intended area. The native speed peaks were 1.10 rad/s across J1–J3, 2.70 rad/s across J4–J6 and 0.136 m/s across the fingers. Efforts stayed within 27/27/27/7/7/7 N·m at the arm joints and 22 N per finger, with a measured finger peak below 13 N. [Measurements and controls](VALIDATION.json) · [Final joint inspection](joint-after.png)

The clips, metal rest, iron, solder pen and exposed copper have native contact. The two hand tools are free bodies. Opposing gripper contacts must act on each tool during feed. A two-node heat model follows force-dependent iron contact and conduction through the touched conductors. Solder feed requires loaded simultaneous contact on the aligned conductors and temperatures above the alloy liquidus, with a passive shortening coordinate. The joint gains a bounded cohesive swivel reaction only after cooling. A simulated fillet illustrates the qualifying modeled feed. The bond is retained and the overlap checked at the end. The simulator's contact and retention checks are proxies for the separate physical continuity and proof-pull assessment.

The unrestrained cable portions are simplified as visual drapes outside the working ends. The iron model uses its working part and grip at the measured class of mass and size, with a shorter rear and without cord dynamics. The thermal constants and contact force limits for the specific metal, flux and finish remain estimates, the solder and thermal transport are approximate, and physical time to wetting, joint strength and heat damage have not been measured. [Lead-free alloy information](https://www.aimsolder.com/aimsolderproducts/alloys-bar-solder/)

## Assets and licenses

- The motor, workholding fixture, bench instruments and wire geometry are original procedural assets, supplied with the task. Product photos are references and are absent from the assets and recordings.
- The [pinned Dream Labs wheel](https://pypi.org/project/dreamlabs/1.0.23/) supplies `yam_env.py`, the oak surface texture, `sim-eval-yam.json` camera calibration and `model.json` hierarchy under its [Apache-2.0 license](licenses/Dream-Labs-Apache-2.0.txt) and [notice](licenses/Dream-Labs-NOTICE.txt). The task retains the unlowered optical poses from [the reference calibration](https://github.com/ARISE-Initiative/yamlab/tree/ec0455d2b4ce35f21fc126418ea5e74ac567133d).
- The same wheel supplies the I2RT YAM URDF, decomposed contact geometry `collisions.npz` and the eleven rendered robot meshes `base_link_visual.stl`, `gripper.stl`, `link_1_visual.stl`, `link2_black.stl`, `link2_white.stl`, `link3_black.stl`, `link3_white.stl`, `link_4_visual.stl`, `link_5_visual.stl`, `tip_left.stl` and `tip_right.stl`. [I2RT MIT license](licenses/I2RT-MIT.txt) · [Upstream third-party notices](licenses/Dream-Labs-third-party.md)

## Board comparison

A search of 4,401 board records found related work on fitting wire to a terminal lug, aligning loose wire tips for a twist-on connector and heat shrinking a previously crimped lug. Each tests a different connection method or phase. This splice starts with two separate conductors and scores creating a cooled soldered join. The board-level capacitor task uses planted leads and through-hole pad wetting. It does not exercise wire lap formation and retention. [Wire to terminal lug](https://tasks.robocurve.org/board?task=TK-CZV7D2) · [Twist-on connector](https://tasks.robocurve.org/board?task=TK-HGKRZN) · [Crimp-lug heat shrink](https://tasks.robocurve.org/board?task=TK-X5ZRTM)
