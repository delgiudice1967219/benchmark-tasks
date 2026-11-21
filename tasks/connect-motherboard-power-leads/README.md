# Connect both motherboard power leads

Connect the power supply’s 24-pin main lead and paired 4+4-pin CPU lead to the motherboard. Leave both connectors fully seated and latched when you let go.

Both leads are required to bring up a compute workstation before chassis installation. The wide main plug and compact two-piece CPU plug have different keys and different original snap catches. A fixed power supply and its two tethers constrain the grasp and release. The benchmark compares two keyed, retained board connections with the plug-in and RJ45-coupler tasks in DataCenterBench. [Intel workstation assembly procedure](https://www.intel.com/content/www/us/en/gaming/resources/how-to-build-a-gaming-pc.html) · [DataCenterBench method](https://robocurve.org/datacenterbench/methods)

## Setup and reset

Use the calibrated dual-YAM station with a 110 × 130 cm wood worktop, 76 cm worktop height and 61 cm base spacing. Set an Open Benchtable V2 with its 360 × 270 × 8 mm aluminum deck, stock 115 mm feet and 25 mm brass motherboard standoffs on the worktop. Its center is 44 mm toward the arm on the world-negative-X side and 201 mm forward from the line of the bases. Keep the stock motherboard and nearby components horizontal. [Open Benchtable specifications](https://openbenchtable.com/specifications/) · [Stock assembly guide](https://openbenchtable.com/how-to/obt-assembly-guide-v2/)

Mount the 244 × 240 mm ASUS Prime B450M-A II FR4 motherboard on its standoffs. Leave its ASUS rear ports, RAM sockets, processor socket, chipset sink, capacitors and surrounding components in the scene. Secure the externally disconnected Corsair CX550 power supply to the bench under the deck. Its 600 mm insulated 24-pin and pre-paired 4+4-pin leads keep their molded nylon housings, 4.2 mm cavity pitch and original latches. The main housing is approximately 51.6 mm wide and the CPU pair is approximately 18 mm wide. [Board layout](https://dlcdnets.asus.com/pub/ASUS/mb/SocketAM4/PRIME_B450M-A_II/E17196_PRIME_B450M-A_II_UM_WEB.pdf) · [Power supply](https://www.corsair.com/us/en/p/psu/cp-9020277-na/cx-series-cx550-550-watt-80-plus-bronze-atx-power-supply-cp-9020277-na)

Leave the main housing loose on the aluminum deck beside the board. Pair the EPS halves using the housing guides and leave its loose nose down on the bare board area near the CPU socket. Route the surplus by the bench sides as in the setup image. No housing begins inside a header. Keep both grippers fully open in their ready pose. Unplug the supply from AC and use an unpowered board during physical trials. Replace damaged hardware before resetting and keep the leads in their original cable routing. The setup file records the exact simulator reset transforms for the bench, board, empty headers, both loose housings and cameras. [Setup image](setup.png) · [Setup transforms](setup.json)

## Scoring

Score each lead independently and sum the two results. Over the final two seconds, both gripper pinch points must be at least 48 mm from the housing center, and neither gripper may touch its housing or tether.

| Lead score | Required result |
| --- | --- |
| 0.00 | The lead fails both higher tiers, uses an incorrect connector, or shows damage to the housing, key, catch, or board. |
| 0.25 | The correct keyed housing remains engaged. Its center is within 1.6 mm of the header center in the board plane and its angle error is less than 9.17°. All four seam gaps are below 4.3 mm. |
| 0.50 | The housing center is within 1.3 mm, its angle error is below 5.73°, and all four seam gaps are below 1.5 mm. Every original nose is captured behind its shoulder, and the housing stays engaged under a 3 N axial pull. |

Use a feeler gauge at the four corners, the close views to inspect orientation and clip engagement, and a hand force gauge along the insertion axis. The simulator checks native lug-face overlap, housing position and hand contact throughout the released hold. The 3 N virtual gauge runs in a separate forward-only copy after the robot stops. The numerical signed gap lower bound for full seating is −0.9 mm. A housing held only by terminal friction earns no full credit without the original catches captured.

## Model and asset provenance

The simulator uses the original calibrated YAM kinematics, contacts, table, overhead and wrist cameras. The controllable force ceilings are 27 N·m at each proximal actuator, 7 N·m at each wrist actuator, and 40 N at each finger actuator. These caps are model limits. The board, bench, contacts, shells and cables are procedural assets created for the task. Native contacts resolve key and latch interaction, and ball joints and a molded strain-relief connection represent the exposed tether. Permanent material and electrical damage are outside the model and require physical inspection.

Each circuit has two internal spring tabs pressing transversely on the header pin. A nominal 1.47 N per tin-plated circuit and 0.50 N nominal axial drag are modeling assumptions based on two separate manufacturer minimums. The resulting friction coefficient, cable forces and full assembly insertion curve have yet to be measured on hardware. The released pose and 3 N measurement are verified in simulation. [Molex mechanical specification, sections 7.2.1 and 7.2.9](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/productspecificationpdf/555/5556/PS-5556-001-001.pdf) · [Molex terminal drawing](https://www.molex.com/content/dam/molex/molex-dot-com/products/automated/en-us/salesdrawingpdf/555/5556/039000077_sd.pdf)

The calibrated workstation generator comes from the `dream_examples/yam_mujoco` example in Dream Labs `dreamlabs==1.0.23`, distributed under Apache-2.0. The I2RT YAM robot meshes and descriptions retain the MIT terms in `LICENSE.yam`. Optical calibration follows the MIT-licensed YAML Lab reference at the pin named in `ASSETS.json`. The wheel SHA-256 and individual referenced asset files are listed in the environment and asset manifests. The dependency declares CC BY 4.0 attribution for the unused example fruit scans, whose bodies are removed from the task scene. The task's motherboard, power supply, connectors and tools use procedural models. [Asset manifest](ASSETS.json) · [Dream Labs source](https://github.com/dreamlabs-co/dreamlabs) · [Dependency package](https://pypi.org/project/dreamlabs/1.0.23/) · [YAML Lab license](https://github.com/ARISE-Initiative/yamlab/blob/ec0455d2b4ce35f21fc126418ea5e74ac567133d/LICENSE)

## Reproduce

Use the Python, MuJoCo and library versions in `ENVIRONMENT.json`. Install `dreamlabs==1.0.23` as an asset dependency. The example also works from its extracted directory by setting `ROBOCURVE_YAM_EXAMPLE_DIR` to the directory containing `yam_env.py` and its `assets` folder. FFmpeg is required to write the review video.

```sh
python -m pip install --no-deps dreamlabs==1.0.23
python episode.py --out results/scripted --record
python dream_power.py --out results/offline-check --dry-run
python dream_power.py --out results/dream-1
```

The simulator executes 2,100 actions at most, at 20 Hz. The prerecorded tape contains only the 14 native joint and jaw targets, with physics stepped at 1,000 Hz. Replay executes the tape from the same complete reset. The independent proof loads do not enter the robot episode or its observations. The three calibrated views and the fixed detail camera are recorded once per control step.

The scripted demo scores **1.00** in 1,412 actions, or 70.6 simulated seconds. Both original catches hold their keyed housings under the separate 3 N tests. Two fresh replays of the packaged tape give identical scores and final measurements. The tape produces 0 N of arm-to-arm contact. A right fingertip contacts the modeled metal rear I/O housing during EPS release, with a peak load of 11.15 N. This contact and the insertion loads remain unmeasured on physical hardware. [Validation](VALIDATION.json)

Dream-1 completed 2,100 actions and 250 inference requests from the same reset at 20 Hz, and scored 0.00. The package uses the official Inspect Robots policy and the three standard robot views. [Validation record](VALIDATION.json)
