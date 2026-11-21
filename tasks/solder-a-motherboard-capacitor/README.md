# Solder both leads of a replacement motherboard capacitor

Solder both leads of the replacement capacitor to the motherboard. Return the iron and solder tube to their holders.

Component-level repair can restore a board by replacing the failed part. Here the faulty capacitor is already removed and its replacement dry-fitted. The trial starts with two unsoldered plated through-holes and ends with two distinct bonded joints. The operator has to control heat and solder delivery within 5 mm of a second electrically separate land. [IPC rework and repair standard](https://www.ipc.org/TOC/IPC-TOC-7711-21C.pdf) · [Commercial board-repair workflow](https://ohmz.technology/services/component-level-board-repair/?lang=en) · [NASA through-hole hand-soldering workbook](https://s3vi.ndc.nasa.gov/ssri-kb/static/resources/NASA%20Student%20Handbook%20for%20Hand%20Soldering.pdf)

## Setup and reset

Use the 110 × 130 × 6 cm wood-top station with its top 76 cm above the floor. The two fixed YAM bases are 61 cm apart, both arms at the ready configuration and both grippers open 95 mm. Use the calibrated overhead and wrist cameras and a fixed oblique judging camera. [Coordinates and fixtures](setup.json)

Use an ASRock P4S55FX+ desktop motherboard, 305 × 218 × 1.6 mm, on a static-safe edge holder. Disconnect power and remove the battery. Keep the solder side up, 106 mm above the table, with the prepared service site nearer the arms. Hold both board edges. Orient the stock capacitor part number EKZE6R3ELL102MJC5S with its marked negative stripe toward the negative site. Its aluminum body measures 10 × 12.5 mm and its two leads are 0.60 mm at 5 mm pitch. Trim the prepared leads to 6.4 mm. The face gap is at most 1 mm. The adjustable insulating support has a shallow 10.4 mm bore underneath the capacitor body to keep the loose part seated while the board is upside down. The setup drawing defines the nominal service location. Measure the corresponding prepared holes and nearby clearances on the stock board before running the physical setup. An equivalent sacrificial desktop board may be used if it has clean plated holes on 5 mm centers, a matching 0.60 mm lead pair, and a representative populated region with the prepared site 10–12 mm from the nearest board edge. Hold the same orientation, board-face height and tool coordinates. [Board manufacturer](https://download.asrock.com/Manual/P4S55FX.pdf) · [Chemi-Con part specifications](https://www.chemi-con.co.jp/en/products/detail-condenser.php?part_number=EKZE6R3ELL102MJC5S)

Use a Weller WEP70 iron with an ETA 1.6 mm chisel tip, parked horizontally by its insulated handle in a safe steel holder. The other holder supports an approximately 110 × 14 mm tube of 0.81 mm SAC305 flux-core solder with 16 mm exposed. Use working shop fume extraction, a heat-resistant mat and tip containment. Keep the iron cable free of robot reach. Set the real iron per the board, part and alloy specifications under trained supervision. [Weller station](https://www.weller-tools.com/we1010na/) · [Tip specifications](https://www.weller-tools.com/sites/default/files/products/documents/WEL_DATA_ET5SetEU_EN.pdf) · [MG Chemicals SAC305 solder](https://mgchemicals.com/products/soldering-supplies/solder-wire/lead-free-solder/) · [Chemi-Con solder guidance](https://www.chemi-con.co.jp/en/catalog/pdf/al-e/al-sepa-e/001-guide/al-precaution-e-2020.pdf)

For each trial, use a new component and prepared, clean, unwetted holes. Restore both tools and the part to their exact starting positions. Verify that the copper lands and separate leads are visible under the fixed cameras. Soldering is the stage under test. Clean the board and trim excess lead length as appropriate after the scored trial.

## Score

Use the full trial video and a magnified final photo. A qualifying joint requires simultaneous contact of the hot tip and the solder supply at its lead and land while a different gripper holds each tool. It has a cooled solder fillet that wets the lead and land. Inspect both leads at their correct polarity and through their own holes. A bridge, detached part, incorrect polarity, heat damage, hot tip striking the arms or table, or a strike between the arms scores 0 and remains invalid across the trial. A cooled fillet at a single connection qualifies independently of the other connection.

| Score | Outcome |
| --- | --- |
| 0 | Neither joint qualifies or a failure applies. |
| 0.50 | One joint qualifies. |
| 0.75 | Both joints qualify. |
| 1.00 | Both qualifying joints remain seated and intact while both tools rest in their holders. Both grippers clear the tools and their pinch points are at least 80 mm away from the tools and capacitor throughout the final second. |

The MuJoCo grader requires force-bearing contacts at 1 kHz, simultaneous bilateral contact on both separate tool handles, and seated lead geometry. It uses a thermal estimate with 217 °C liquidus and requires a cumulative 0.81 mm³ solder proxy per joint. Final return requires each tool center within 10 mm of its registered holder, each shaft within 16.3° of its start axis, and the tools still. The pose, contact, dose and safety telemetry are in `trace.jsonl`. The rule transfers material only at a wetted hot lead/land after contact. A static simulated force or temperature threshold is an implementation of the physical joint check. Physical grading uses wetting and the recorded manipulation.

## Reproduce

Use Python 3.12 and the pinned dependencies in [ENVIRONMENT.json](ENVIRONMENT.json). Meshes and calibration load from the Dream Labs example dependency. The video is captured on the CPU through Mesa EGL and encoded with FFmpeg at the native 20 Hz control cadence. The action limit is 1,000 actions or 50 simulated seconds.

```sh
python -m pip install 'numpy==2.3.1' 'scipy==1.17.0' 'mujoco==3.11.0' 'opencv-python-headless-noffmpeg==4.14.0.94rc1'
python -m pip install --no-deps 'dreamlabs==1.0.23'
python run.py --replay commands.npy --out runs/full
python render.py runs/full
python run.py --mode cold --staged --replay commands.npy --out runs/cold
python run.py --mode idle --staged --out runs/idle
python run.py --mode one-joint --staged --out runs/one-joint
```

The full replay starts from an exact deterministic reset. Each entry of `commands.npy` is a pair of six joint positions in radians plus normalized jaw fraction, beginning with the ready configuration. The file contains 754 entries and issues 753 actions. No object state or solver telemetry is consumed by replayed control. The development planner in `run.py` uses known simulator geometry and inverse kinematics. Native full, cold, hands-off and one-joint trials verify the same contact grader. See [Recorded measurements](VALIDATION.json).

With the installed policy packages, `--dry-run` checks both action and camera integration on eight holds without network calls. The model receives the initial instruction, calibrated three RGB views at 320 × 176 and the measured joints only. The live runner uses the same full trial reset, clock and scoring implementation.

```sh
python -m pip install 'inspect-robots==0.60.0' 'inspect-robots-yam==0.37.0' 'dreamlabs-client==1.0.14'
python -m pip install 'inspect-robots-dream @ git+https://github.com/kairosiann/inspect-robots-dream.git@b5e936a201cec66c54c297e522c920d98445e339'
python dream_capacitor.py --dry-run --out runs/offline-policy
python dream_capacitor.py --out runs/dream
```

Use `DREAM_API_KEY` or the SDK credential store for the live call. Media use the four-view layout with a fixed detail camera, overhead and both wrists. Install FFmpeg and Mesa EGL via the operating system. `ROBOCURVE_DREAM_YAM` can point to an extracted copy of the pinned wheel's `dream_examples/yam_mujoco` directory.

## Fidelity and provenance

The fixture, plated holes, component, leads, iron and dispenser use procedure-specific geometry with native contacts. The two passive hinge leads deflect under contact. Fillet acquisition depends on heating and feeding the active connection with both tools held. Cooling both joints adds a constraint at the actual settled component transform. Material calibration, solder flow, tip shape, cord handling, the complete stock board topology and electrical integrity are approximate or outside the simulator. The board envelope, part number, through-hole lead pitch and visual evaluation follow a documented hardware repair approach. The nominal fixture location and thermal parameters require physical calibration before claims about exact production tolerances.

- The scene objects, tools, score, trajectories and images are authored for this task. The procedural board uses the manufacturer's dimensions without copied photos or logos.
- Calibration, wood texture and station example come from [Dream Labs 1.0.23](https://pypi.org/project/dreamlabs/1.0.23/). The [Apache 2.0 license](licenses/Dream-Labs-Apache-2.0.txt), [distribution notice](licenses/Dream-Labs-NOTICE.txt) and [third-party terms](licenses/Dream-Labs-third-party.md) identify the dependency. The stock arm, wrist and gripper geometry is from I2RT's [YAM model](https://github.com/i2rt-robotics/i2rt) under [MIT](licenses/I2RT-MIT.txt).

## Board comparison

This task requires two real heat-and-feed joints on opposite leads of one dry-fitted through-hole device, without bridging the nearby lands. A related LED task ended at holding a part with a cold tool contact. Joining two loose wires separately tests aligning and stabilizing conductors in free space. The fixed, dense pads and successive connections are different here. The reference is a legacy desktop motherboard and the task is general electronics repair, with no claim about an AI hardware workflow or the number of workers performing this operation. [LED task decision](https://tasks.robocurve.org/board?task=TK-M33ZRJ) · [Robocurve review prompt](https://tasks.robocurve.org/review-prompt)

Dream-1 completed all 1,000 actions and 113 inference requests from the same initial native state as the scripted replay. It scored 0.00 and left both joints unwetted.
