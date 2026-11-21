# Repair a torn workbook page

Repair the torn workbook page and leave it turned over, with the print visible.

Repairing an ordinary workbook page keeps the printed exercise legible for the next use. This task tests aligning two flexible edges, applying a pressure-sensitive repair while the paper stays aligned, and turning the joined page. [Alaska State Library book repair manual](https://library.alaska.gov/documents/hist/convservation-manual.pdf) · [Neschen filmoplast P technical information](https://www.neschen.de/wp-content/uploads/2016/12/Technical-Information-filmoplast%C2%AE-P.pdf)

**Category** Societal relatability. Use ordinary dry, replaceable workbook paper.

The 21 November board review identified TK-9RHJXM, drywall tape, and TK-HYQRVJ, a shipping-carton tape gun, as the closest listings in the audit. In this task both parts of the torn sheet can deflect, the print must stay legible across the seam, and the final repair must hold as the page turns. [Drywall tape task](https://tasks.robocurve.org/board?task=TK-9RHJXM) · [Carton taping task](https://tasks.robocurve.org/board?task=TK-HYQRVJ)

## Setup and reset

Use the fixed dual YAM station with robot bases 610 mm apart and its original laminate table at 760 mm. Start both grippers open in the station's ready pose.

- Use one open A5 arithmetic workbook with a 210 × 148 mm, approximately 0.14 mm thick paper page on the side facing the robot bases. Set the spine parallel to the line between bases. Its midpoint is 25 mm to the right and 458 mm toward the detail camera from the midpoint between bases. Use a hardcover workbook with paper bookcloth or laminated card covers and approximately 15 mm of sheets on each side. Place the open book on a flat 222 × 284 × 31 mm wooden support. The working paper lies at 813 mm.
- Make a 66 mm tear running inward from the middle of the free edge, roughly perpendicular to the spine. Both flanks remain attached to the inner page. Give one side of the tear a light upward curl. Its free edge starts 7–9 mm higher than the other side. Use a fresh page with an equivalent tear for each repeat.
- Provide a 20 × 74 mm strip of clear document-repair tissue with a release liner. Hold the liner on a weighted 72 × 92 mm desk clip block, its top 37 mm above the table. Extend the tissue about 8 mm past the exposed end of the liner for pickup. The strip's starting center is 212 mm to the left and 337 mm toward the detail camera from the midpoint of the bases, and its long axis is perpendicular to them. Hold the other end of the liner under the spring clip. Use a fresh liner and strip at every reset.

Directions are as seen from the detail camera facing the robot bases. Use a low-value workbook with printed lines crossing the tear. Replace any wrinkled or previously taped setup pages and restore the starting curl and position before every run. Bench supplies need no power or sharp objects during the trial. The submitted media includes the reset image.

## Scoring

Use a close fixed video and the overhead view. A valid repair covers the seam from within 4 mm of its inner end through within 6 mm of the free edge. The transparent strip must bridge both flanks along the last three quarters of its length. Check that the print remains readable. Along the whole tear, the flanks may have at most a 2 mm gap in three dimensions, and their edges may neither overlap nor fold over each other. A ruler and two stills from the video are sufficient to check a borderline gap.

- **0.00** when no valid repair is visible in the released final state. This covers new damage to the page or strip, tape left on its liner, holding the torn edges together by hand, or tape stuck wholly to one flank.
- **0.40** for an intact repair left released for the last two seconds, with a visible half-second released inspection while the page remains on its original side.
- **1.00** when after that released inspection the page is turned by gripping just one flank, the repair remains intact through the turn, and the whole page rests on the other side at 165° or more for the final two seconds clear of robot contact.

The simulation tracks adhesive contact regions by physical loads and measures flap gaps and strip registration in the page's frame, including through the turn. The 680-action cap provides 34 seconds at 20 Hz.

## Reproduction

Use the installed DreamLabs 1.0.23 YAM MuJoCo example station. Dependencies include NumPy, SciPy and OpenCV, with ffmpeg for the four-view video. The policy adapter also requires inspect-robots, inspect-robots-dream and inspect-robots-yam.

    export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
    export MESA_SHADER_CACHE_DISABLE=true
    python replay.py --out runs/replay --video runs/replay.mp4 --setup runs/new-setup.png
    python replay.py --place-only --out runs/partial
    python replay.py --one-flank --out runs/one-flank
    python replay.py --skew --out runs/skew
    python replay.py --hold --out runs/hold
    python dream_run.py --dry-run --limit 2 --out runs/offline-policy-check

Each scripted run records joint and jaw commands at 20 Hz, the contact and registration trace, and a JSON summary. The commands are preset geometry-based waypoints and inverse kinematics for the stock arms. The simulator handles liner peel, paper motion and bonding from contacts during each ordinary physics step.

The scripted score was **1.00** in **679 actions**. The page settled at 179.08° and the maximum measured seam gap was 0.53 mm at release. The flat released patch scored 0.40. The one-sided, diagonal placement and sustained-grasp controls scored 0.00. The live Dream-1 trial scored **0.00** in **680 actions** across 73 inference requests. The submitted review media contains both trials. [Validation](validation.json)

## Fidelity and assets

The page has an intact bound section and two thin independently hinged paper flanks with a stepped, complementary seam. The rigid segments approximate paper flex at the tear. The clear tissue is four thin, articulated sections. Liner anchors release in order under robot-induced peel tension. Adhesion constraints appear at matched surface contacts after sustained native pressure. The material updates change only these local attachment properties. Paper grain, tearing growth, adhesive residue and long-term stability are outside this prototype. The local contact margins, force thresholds and real YAM reach have yet to be validated on physical hardware.

- Paper, tape, liner, clipboard and booklet graphics are authored procedurally here.
- The installed stock YAM station and meshes are supplied by DreamLabs. The robot geometry is credited to I2RT Robotics under MIT. [I2RT notice](LICENSE.i2rt) · [I2RT Robotics](https://github.com/i2rt-robotics/i2rt)
