# Repair a torn workbook page

Repair the torn workbook page and leave it turned over, with the print visible.

Repairing an ordinary workbook page keeps the printed exercise legible for the next use. This task tests aligning two flexible edges, applying a pressure-sensitive repair while the paper stays aligned, and turning the joined page. [Alaska State Library book repair manual](https://library.alaska.gov/documents/hist/convservation-manual.pdf) · [Neschen filmoplast P technical information](https://www.neschen.de/wp-content/uploads/2016/12/Technical-Information-filmoplast%C2%AE-P.pdf)

**Category** Societal relatability. Use ordinary dry, replaceable workbook paper.

The 21 November board review identified TK-9RHJXM, drywall tape, and TK-HYQRVJ, a shipping-carton tape gun, as the closest listings in the audit. In this task both parts of the torn sheet can deflect, the print must stay legible across the seam, and the final repair must hold as the page turns. [Drywall tape task](https://tasks.robocurve.org/board?task=TK-9RHJXM) · [Carton taping task](https://tasks.robocurve.org/board?task=TK-HYQRVJ)

## Setup and reset

Use the fixed dual YAM station with robot bases 610 mm apart and its original laminate table at 760 mm. Start both grippers open in the station's ready pose.

- Use one open A5 arithmetic workbook with a 210 × 148 mm, approximately 0.14 mm thick paper page on the side facing the robot bases. Set the spine parallel to the line between bases. Its midpoint is 25 mm to the right and 458 mm toward the detail camera from the midpoint between bases. Use a hardcover workbook with paper bookcloth or laminated card covers and approximately 15 mm of sheets on each side. The stack below the working page ends 4 mm short of its free edge, leaving room to grip the edge. Place the open book on a flat 222 × 284 × 31 mm wooden support. The working paper lies at 813 mm.
- Make a 66 mm tear running inward from the middle of the free edge, roughly perpendicular to the spine. Both flanks remain attached to the inner page. Give one side of the tear a light upward curl. Its free edge starts 7–9 mm higher than the other side. Use a fresh page with an equivalent tear for each repeat.
- Provide a 20 × 74 mm strip of Neschen filmoplast P transparent repair tissue on its silicone-coated glassine liner. Hold the liner on a weighted 72 × 92 mm desk clip block, its top 37 mm above the table. Extend the tissue about 8 mm past the exposed end of the liner for pickup. The strip's starting center is 212 mm to the left and 337 mm toward the detail camera from the midpoint of the bases, and its long axis is perpendicular to them. Hold the other end of the liner under the spring clip. Use a fresh liner and strip at every reset.

Directions are as seen from the detail camera facing the robot bases. Use a low-value workbook with printed lines crossing the tear. Replace any wrinkled or previously taped setup pages and restore the starting curl and position before every run. Bench supplies need no power or sharp objects during the trial. The submitted media includes the reset image.

## Scoring

Use a close fixed video and the overhead view. A valid repair covers the seam from within 4 mm of its inner end through within 6 mm of the free edge. The transparent strip must bridge both flanks along the last three quarters of its length. Check that the print remains readable. Along the whole tear, the flanks may have any gap from 0 to 2 mm in three dimensions. Their edges must stay in order without overlapping or folding. A ruler and two stills from the video are sufficient to check a borderline gap.

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
    python replay.py --place-only --light-press --out runs/light-press
    python replay.py --peel-only --no-grip --out runs/liner-control
    python replay.py --one-flank --out runs/one-flank
    python replay.py --skew --out runs/skew
    python replay.py --hold --out runs/hold
    python -m unittest test_revision
    python dream_run.py --dry-run --limit 2 --out runs/offline-policy-check

Each scripted run records joint and jaw commands at 20 Hz, the contact and registration trace, and a JSON summary. The commands are preset geometry-based waypoints and inverse kinematics for the stock arms. The simulator handles liner peel, paper motion and bonding from contacts during each ordinary physics step.

The revised scripted run scores **1.00** in **679 actions**. The page rests at 179.39° with an approximately 0.52 mm maximum gap in the released seam. The flat released patch scores 0.40. Light pressing forms zero bonds. The light-press, one-sided, diagonal placement, ungripped peel and sustained-grasp controls score 0.00. A fresh Dream-1 run on this revision scored **0.00** in **680 actions** across 73 inference requests. [Validation](validation.json)

## Fidelity and assets

The page has an intact bound section and two thin independently hinged paper flanks with a stepped, complementary seam. The rigid segments approximate paper flex at the tear. Four articulated sections represent the transparent tissue. Paper grain, tearing growth, residue and long-term aging are omitted.

Neschen specifies a 54 µm glassine liner and a release-force range of 20–45 mN per centimetre of width at 300 mm/min. For the 20 mm strip the range is 0.040–0.090 N. The simulator uses the 0.065 N midpoint for upward tension at the actively opening front. The scripted opening movement starts near 5 mm/s. The three simulated release forces are 0.0670, 0.0690 and 0.0653 N. The simulator uses a fixed threshold for the duration of the peel. [Filmoplast P technical sheet, pages 2–3](https://www.neschen.de/wp-content/uploads/2016/12/Technical-Information-filmoplast%C2%AE-P.pdf)

Neschen describes firmly rubbing the tissue down across the repaired tear. For a firm local press reference, the model scales the line load of a 2 kg, 45 mm-wide adhesive-test roller to a 2 mm fingertip-edge sample. This gives an application force of 0.872 N per region. Immediate bonding on the workbook paper is an engineering approximation. The application load for this tissue and paper has yet to be measured. [Neschen repair catalog, page 39](https://neschen.es/sites/default/files/2024-10/Neschen.pdf#page=39) · [Adhesive-test roller specifications](https://www.jarp.nl/products/handroller-finat/)

The simulation projects native finger and paper reactions onto the tissue surface normal for each section and flank. It requires a mean downward finger force of at least 0.872 N within 0.10 s. A sample contributes only when that same region also carries at least 0.30 N from the paper beneath it. The paper threshold is a simulation assumption to verify direct support while the load spreads through the layered contact. Finger samples below 0.218 N contribute zero and spikes contribute at most 1.744 N. A bond forms only while the region is still under supported compression. The inner end adheres to the intact page and the remaining three sections adhere to both flanks. The seven scripted bonds record simultaneous paper reactions from 0.387 to 1.499 N at attachment.

- Paper, tape, liner, clipboard and booklet graphics are authored procedurally here.
- The installed stock YAM station and meshes are supplied by DreamLabs. The robot geometry is credited to I2RT Robotics under MIT. [I2RT notice](LICENSE.i2rt) · [I2RT Robotics](https://github.com/i2rt-robotics/i2rt)
