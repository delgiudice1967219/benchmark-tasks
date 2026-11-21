# Replace a DualSense Edge stick module

The robot switches a faulty stick module for a loose spare and returns the
unpowered controller to a ready-to-use mechanical state. The full sequence
starts with the original stick module and front cover in place. A loose
controller means the arms must stabilize it while operating its small release
and work around both thumbsticks when fitting the cover.

## Status

The fixed-joint replay scores 1.00 from the untouched reset in 2,212 actions
(110.60 seconds at 20 Hz). The arms open and remove the cover, exchange and
lock the module, and secure the cover with a final contact-guided wrist turn
while pressing the seam. The score stays valid with both hands clear for
3.59 seconds. The 2,400-action hands-off control scores 0.00. Dream-1 completed 2,400 actions and 288 inference requests from the same reset. It scored 0.00 and left the front cover installed.

The board media include the 20 fps four-view replay and reset image. [Independent replay grade](evidence/full-score-repeat/summary.json) · [Hands-off grade](evidence/hands-off/summary.json)

## Real task

Use one powered-off, unplugged Sony DualSense Edge controller
(CFI-ZCP1, about 160 by 106 by 66 mm and 325 g) and one compatible spare stick
module (about 30 by 40 by 31 mm and 15 g). The controller uses a rigid plastic
shell and plastic front cover, rubberized caps, a metal release lever, and
electrical contacts. Use the stock shell, front cover, and release mechanisms.
Match the setup photo. Put the controller on its front with the grips toward
the arms, its center 34.3 cm in front of the arm-base line and 2.2 cm toward the
left arm from the midpoint. Set the spare upright 31.4 cm in front of the base
line and 9.8 cm toward the right arm from the midpoint. The arm bases are
61 cm apart. The rigid laminate table is 110 by 130 cm with the tabletop
76 cm above the floor, 6 cm thick. Both YAM grippers start open in the ready
pose. The fixed overhead and two wrist cameras observe the trial.

Reset by returning the original stick to its guide and lowering the release
lever. Refit and check the front cover. Place the spare upright at its starting
location, turn the controller over and return both arms to ready. The controller
stays off with the USB cable disconnected. Follow Sony’s static-discharge and
connector-handling instructions.

[Official replacement guide](https://www.playstation.com/en-gb/support/hardware/edge-replace-stick/) · [Official controller manual, pp. 14–15 and 20](https://www.playstation.com/content/dam/global_pdc/en/corporate/support/manuals/accessories/ps5-accessories/dualsense-wireless-controller-edge/CFI-ZCP1_PS5_Wireless_Controller_Instruction_Manual%24en.pdf)

## Scoring

Use the highest completed tier. Review the entire trial for drops or damage. Check the original module from its guide through its placement beside the repaired controller. Use the fixed detail video and a ruler or a calibrated final photograph for the lever angle, module seam, cover fit, and gripper clearance.

| Score | Outcome |
| --- | --- |
| 0.00 | No milestone qualifies, a part is damaged, or a part falls off the tabletop. |
| 0.20 | The rear latch releases and the front cover separates at least 42 mm from its fitted position. |
| 0.45 | The retaining lever reaches 59° and the original module travels out along its guide to at least 64 mm from the socket. |
| 0.75 | The replacement moves along the same guide and seats within 2 mm of the stock stop. |
| 1.00 | The lever is closed, the cover remains secured within 1 mm of the reference seam, the controller rests face up, the original module rests separately, and both pinch points are at least 90 mm clear for the final second. |

## Simulation

Replay the recorded joint targets from a new reset.
The local copy of the standard station comes from the Dream Labs SDK example
`dreamlabs==1.0.23`. The replay used MuJoCo 3.11.0, NumPy 2.3.1, and SciPy
1.17.0. The optional video uses OpenCV, FFmpeg, and Mesa EGL. The shared
`review_media.py` helper included in this folder composes the 20 Hz detail,
overhead and two wrist views. The calibrated overhead and both wrists are
the policy observations.

```sh
python run.py --record --out evidence/replay
python run.py --mode hands-off --out evidence/hands-off-repeat
python dream_dualsense.py --dry-run --out evidence/policy-check
```

The controller, ABS-like front cover, replacement and original modules, and
sprung controls use procedural geometry. The workbench and both stock YAM arms
use standard link and fingertip collision meshes. The gripper carries and
aligns free parts through physics contacts. The cover clip uses an explicit
approximation of a small sprung plastic/metal latch. The robot has to drive
the native rear latch, physically remove and align the cover, and press the
seam within the closed pose tolerance before the clip can retain the measured
pose. The clip does not move the cover into alignment. The
standalone model approximates internal dimensions, clearances, friction and
spring loads. The module guide and metal lever use rigid collisions, Coulomb
friction and a passive constrained hinge. The model does not simulate
connector continuity, plastics fracture, or electrostatic damage. The
dimensional approximations and forces have no physical bench validation. The simulated fingers exceeded the 0.146 m/s nominal speed in one 50 ms control interval during cover transfer. The observed interval mean was 0.282 m/s, with a 0.558 m/s physics-substep peak.

For the official Dream-1 assessment, use `dream_dualsense.py --attempt` with a
new output directory and a configured `DREAM_API_KEY`. The service receives
the task instruction, three RGB views, and robot joint positions. The
`--dry-run` command runs the camera and control path locally with zero service
requests. [Pinned runtime](ENVIRONMENT.json) · [Validation records](VALIDATION.json)

## Asset notices

- YAM arm and gripper visual meshes, collision hulls, inertials, URDF, gripper
  layout, and calibration come from the `dream_examples/yam_mujoco` station in
  Dream Labs SDK 1.0.23. The station includes the I2RT Robotics MIT asset
  notice. [I2RT MIT notice](LICENSE.i2rt)
- The station tabletop texture is the SDK's bundled `assets/oak.png`. Install the pinned [Dream Labs station dependency](https://pypi.org/project/dreamlabs/1.0.23/). [Apache-2.0 license](licenses/Dream-Labs-Apache-2.0.txt) · [Distribution notice](licenses/Dream-Labs-NOTICE.txt) · [Third-party notices](licenses/Dream-Labs-third-party.md)
- Controller, cover, lever, module, and their colors are original procedural
  geometry. The Sony hardware manual informed overall size and motion. Sony
  photographs, logos, or illustrations are absent from the simulation.

## Board comparison

The rail segment shares keyed translation with the accepted linear-guide carriage. The full repair also requires releasing a rear latch on a free controller, removing a separate cover, managing a second retaining lever, and refitting the cover around two installed thumbsticks. [Accepted rail task](https://tasks.robocurve.org/board?task=TK-UKX5MH) · [Joy-Con task decision](https://tasks.robocurve.org/board?task=TK-VM869E)
