# Spread a suit of playing cards for inspection

Spread the thirteen face-up hearts across the felt in one continuous overlapping row, keeping Ace through King in order and every rank readable. Leave the row untouched for two seconds.

A face-up spread lets another person check that the cards of a suit are all present before play. The opening-table procedure for Maryland Run 'Em Twice Poker calls for a visual inspection of the cards spread in suit and sequence. This task models the spread of one complete suit as a substep of that procedure. The skill is progressively releasing adjacent layers of one packet during a moving contact stroke. [Maryland opening-table procedure, chapter 4](https://www.mdgaming.com/wp-content/uploads/2025/02/Run-Em-Twice-Poker-Standard-Rules-Version-1.0.pdf) · [Roberto Giobbi's ribbon spread technique, pp. 98–99](https://s3-us-west-2.amazonaws.com/ibmresources/Roberto_Giobbis_Introduction_to_Card_Magic_IBM.pdf)

TK-ZPAQZ4 tests drawing and dealing individual face-down cards. FACT's shuffle interleaves two packets. DexHoldem lists individual pick-up, showing and placement actions. The final test here is an intact, ordered ribbon with each rank simultaneously open to view after a single stroke. [Robocurve dealing task](https://tasks.robocurve.org/board?task=TK-ZPAQZ4) · [FACT task 5](https://arxiv.org/html/2610.19155v1) · [DexHoldem primitive definitions](https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot#actions)

## Physical setup and reset

Use the standard fixed dual YAM station with the bases 610 mm apart and the laminate top 760 mm above the floor. Attach a flat 510 × 248 × 2 mm green polyester close-up felt mat to the table so it remains flat during the trial. Set one ordinary 13-card heart suit from a Euro Poker-size coated paper playing-card deck face up on the felt. Nominal cards are 88 × 63 mm, about 0.31 mm thick and 1.8 g each. Stack Ace on the bottom through King on top, all facing the same way. Square the edges within 1 mm, with the long edges parallel to the line between the robot bases. The left-facing corner rank must be at the far edge from the fixed observer. There is no side guide, raised holder or loose tool. [Euro Poker card specifications](https://d1vzi28wh99zvq.cloudfront.net/images/site_resources/Specifications%20for%20Printing%20Poker%20Cards%20Rev.pdf)

Read the setup as seen from the detail camera facing the robot bases. The center of the felt is 100 mm to the left and 338 mm forward from the midpoint of the bases. The packet center is 215 mm to the left and 338 mm forward from the same midpoint. Match the setup image and start both grippers open in the station's ready pose. Re-square the cards and replace any torn or permanently bent card before a new trial.

## Scoring

Use the close fixed video to verify a single continuous sweep of the packet with overlapping release. Moving separated cards one by one into a row, losing a card, or tearing or permanently bending a card scores zero. Once the arms have released, use the overhead image to read the printed ranks. Find the longest uninterrupted run with all its ranks readable face up and in increasing order. Each consecutive pair must overlap by at least one tenth of a card's plan-view area, with their printed rank centers at least 4 mm apart along the spread direction. Use an overhead photo and a ruler if the overlap is close to the boundary.

- **0.25** when an uninterrupted run of at least three ranks is readable.
- **0.50** when an uninterrupted run of at least eight ranks is readable.
- **1.00** when all thirteen are readable in one run, all cards remain on the felt, and both arms stay off the cards for two continuous seconds after setting the spread down. During this hold the tool-center point of each gripper must be at least 55 mm from every card center. Use the close and overhead views to check clearance.
- **0.00** otherwise, including a cleared or squared packet with no valid sweep.

Use the video to check that one gripper spreads the packet in a single pass. Contacting the cards with the other arm or returning to rearrange cards after the sweep invalidates the trial. The simulator monitors contact at every physics step, detects fingertip contact with multiple layers, and tracks the active hand and the packet's motion. The visual grader tests the projected rank patches, overlapping footprints, order, and released dwell. The scripted policy emits joint targets computed from the reset and CAD dimensions, and the method audit runs unchanged for the learned policy.

## Reproduction and policy interface

The scripted replay uses 214 commands at 20 Hz from the standard ready pose. The policy has a 900-action limit, or 45 simulated seconds. Dynamics use MuJoCo at 0.00025 s. This benchmark uses the installed DreamLabs 1.0.23 YAM scene with its stock arm geometry, drives, wrist cameras and overhead camera. The stationary detail camera makes small ranks readable in review. Python dependencies are `mujoco`, `numpy`, `scipy`, `pillow`, `opencv-python`, and `dreamlabs==1.0.23`. The policy interface also requires the installed `inspect-robots`, `inspect-robots-dream` and `inspect-robots-yam` packages. The software-render examples use EGL with Mesa and `ffmpeg`.

```bash
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
export MESA_SHADER_CACHE_DISABLE=true
python replay.py --two-tips --stroke .310 --rise .0049 --height .01012 \
  --delay .07 --duration 2.45 --profile .75 --trace \
  --out replay-score.json --video replay.mp4 --actions replay-actions.npy
python replay.py --idle --out idle-score.json
python replay.py --two-tips --stroke .310 --rise .0049 --height .01012 \
  --delay .07 --duration 2.45 --profile .75 --negative --out clear-sweep-score.json
python dream_run.py --dry-run --limit 2 --out offline-check
```

The no-contact control raises the same joint-planned sweep above the card packet. The other control holds both arms at rest for the entire policy limit. Both score zero. Run `python dream_run.py --out dream1-live` for a live trial of the full policy limit. Set a fresh output directory for each media run.

The recorded Dream-1 episode completed 900 actions and made 100 inference requests. It scored **0.00**, with only the top rank visible and no qualifying fingertip sweep. The recorded scripted trial scored **1.00** after 214 actions. [Trial timings, scores and media hashes](validation.json)

The same scripted commands scored **1.00** from each of two perturbed resets within the 1 mm squaring tolerance. Both trials exposed all thirteen ranks. The method audit also rejected simulated contact events from either arm after the sweep. [Reset and contact checks](validation.json)

## Fidelity and assets

Each card is an independent rigid coated-paper surrogate with native edge, face, hand, and felt contacts. Flat cards match the guiding stroke in this trial. Omissions include card flex, bending damage, and surface-finish variation. The assumed felt/card, card/card, and dry gripper/card friction coefficients are 0.62, 0.22, and 0.43. They have not been measured on physical hardware. For a physical repetition, characterize the felt and coating with a tilt test and inspect the cards before and after the sweep. No artificial release, teleport, backstop, spring, or pose correction moves the cards after reset.

- Cards, mat, rank and court artwork are authored for this task. Texture generation uses DejaVu Sans Bold under the Bitstream Vera license and public-domain DejaVu changes. [Font notice](LICENSE.dejavu)
- Stock YAM meshes and collision geometry are supplied by the installed DreamLabs dependency and credited to I2RT Robotics under MIT. The example code and procedural station are used through the installed dependency. [I2RT notice](LICENSE.i2rt) · [I2RT Robotics](https://github.com/i2rt-robotics/i2rt)
