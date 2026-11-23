# Refill a centre-pull towel dispenser

The portable dispenser begins empty and open. The dry refill lies on the worktop with its cardboard sleeve still inside. The completed task leaves one clean sheet immediately available and the remaining supply contained. [Manufacturer loading guide](https://www.contenthubext.kimberly-clark.com/ePIM/ImageStore/public/202057731.pdf)

The station uses a 1,300 × 1,100 mm worktop at 760 mm height, the stock dual YAMs, and calibrated overhead and wrist cameras. The fixed detail camera shows the central nozzle, latch and workspace. The station frame origin is on the centre line, with stock bases at x = ±305 mm and y = 240 mm. The loose refill and housing face in the +y direction. Their setup coordinates and dimensions are in [task.json](task.json). The dispenser and refill were modeled using the following product data. [Dispenser specifications](https://www.kcprofessional.com/en-gb/products/dispensers/wipe-dispensers/centrefeed-wipes-dispensers/6221/6221) · [Refill specification](https://www.kcprofessional.com/en-gb/products/wiping-and-cleaning/general-cleaning/centrefeed/6223/6223) · [Centre-feed design and mounting orientations](https://patents.google.com/patent/WO1997021377A1/en)

Daily custodial work includes refilling dispensers and checking that towels dispense correctly. Johnston County Public Schools specifies removable-core, perforated centre-pull rolls in its custodial supply tender and estimates 6,000 six-roll cases. The Florida Department of Transportation lists the same refill type in its facility cleaning specification. Supplying restrooms is a core duty for janitors and building cleaners. The U.S. Bureau of Labor Statistics reports about 2.4 million jobs across that occupation in 2025. [Daily Maryland custodial duties](https://dhs.maryland.gov/documents/Request%20for%20Proposal/CRRL-LGA%2020-017-S/CRRL-LGA-20-017-S%20-%20Final.pdf) · [Johnston County specifications, p. 9](https://evp.nc.gov/_entity/annotation/d0004de1-77c9-ef11-b8e9-001dd809bcaf/863ea987-6d3e-ed11-9daf-001dd805ec0b?t=1736035200254) · [Florida facility cleaning specification](https://imlive.s3.amazonaws.com/Florida%20Department%20of%20Management%20Services/ID294988525963408949650863119440142678272/F1610262585_DOTRFP215007JANCOMPLETE.pdf) · [O*NET occupational tasks](https://www.onetonline.org/link/details/37-2011.00) · [Occupational Outlook Handbook](https://www.bls.gov/OOH/building-and-grounds-cleaning/janitors-and-building-cleaners.htm)

The roller-feed priming task TK-M5B4NT starts with a seated roll and uses a lever to feed its edge through a roller nip. This task starts with a loose cored refill and uses a circular nozzle. The robot removes the inner sleeve, threads the folded leader, then dispenses one sheet at its perforation.

## Run the demo

The verified configuration uses Python 3.12.12, MuJoCo 3.11.0, NumPy 2.3.1, SciPy 1.17.0, OpenCV 4.14.0, and the DreamLabs 1.0.23 YAM example assets. Video also needs ffmpeg and a supported MuJoCo render backend. On a Linux node with Mesa and EGL, run from this task directory.

```sh
CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python replay.py actions.npy --out /tmp/towel-scripted --video
```

The action sequence contains only joint and gripper setpoints. The trial starts with the loose refill in its reset position. The renderer writes four views at the original 20 Hz. `summary.json` reports the same score used for the policy run. The reset and action sequence are deterministic and use no random seeds. The verified scripted replay uses 2,876 controls, or 143.8 simulated seconds. The policy limit is 3,000 controls at 20 Hz, or 150 simulated seconds. The native solver runs at 1 kHz.

## Score a trial

Use the fixed camera recording to check the required install and dispensing sequence. Compare the final overhead view and setup dimensions using a ruler. Check the roll's lateral position, insertion depth and axis in a still with the lid open before closing. After the released hold, measure the visible next towel along its folds from the opening. Reopen the lid for a second photo if the final roll position needs confirmation, keeping the housing in place. Measure the removed towel's nearest edge to the case and verify the entire sheet is on the table. The four cumulative thresholds appear in [task.json](task.json), and the runtime scorer is in [score.py](score.py). The highest credited outcome must hold for two seconds with both grippers clear of the task objects. The compressed core and equipment are intact throughout the demonstration.

## Run the policy

The policy uses the same scene, instruction, action limit, views and scoring. The validated interface uses Inspect Robots 0.60.0, Inspect Robots Dream 0.1.0, and Inspect Robots YAM 0.37.0. Supply your Dream-1 access through the installed Inspect Robots client. Keep authentication outside the task directory.

```sh
python dream_run.py --out /tmp/towel-dream --scripted-evidence /tmp/towel-scripted
```

## Simulation fidelity

The physical benchmark uses a dry paper refill and a portable case. The model uses the calibrated stock YAM geometry and physically colliding apparatus. Each 380 mm active paper sheet is represented as an articulated folded strip, with bend yield and a residual crease. A continuous visible surface follows the physical contact links. The first seam releases only after at least 3 N of native tensile load for 22 ms. This threshold, the friction coefficients and local dimensions of the nozzle and catch are model parameters. The product parameters were sourced from manufacturer literature. No physical sample was available for measuring the local contact behavior.

## Assets and licences

- I2RT YAM robot mesh, URDF, and gripper geometry are provided by the DreamLabs YAM example. The original robot assets are MIT licensed. [I2RT source](https://github.com/i2rt-robotics/i2rt) · [License](LICENSE.i2rt)
- The stock tabletop, oak texture, fixture, and camera calibration are referenced from the installed DreamLabs YAM example. The DreamLabs package declares Apache 2.0 for its original contents. The station files are loaded from that dependency. [DreamLabs package](https://pypi.org/project/dreamlabs/1.0.23/)
- The case, latch, nozzle, cardboard, paper, and roll are original procedural task assets. The linked manufacturer documents are design references.
