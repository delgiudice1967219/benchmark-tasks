# Mannequin and harness build dimensions

Use a rigid PETG shell with EVA pads on the chest, leg segments and paws. The dimensions below describe the outer padded surfaces. Use removable ballast to reach 0.78 kg total, with about 0.58 kg in the torso and head assembly and 49 g in each complete leg. The robot grips the padded chest shell. Secure the ballast so it cannot shift.

## Body coordinates

Coordinates are in millimetres from the rigid trunk reference point. Positive X points along the line between the YAM bases, positive Y points toward their side of the table, and positive Z points upward. The muzzle faces negative Y. Each size gives full width × length × height.

| Part | Centre or ends | Size |
| --- | --- | --- |
| Chest ellipsoid | `(0, −26, 0)` | 80 × 172 × 98 |
| Haunch ellipsoid | `(0, 70, −3)` | 70 × 124 × 86 |
| Neck capsule | `(0, −80, 18)` to `(0, −117, 47)` | 56 mm diameter |
| Head ellipsoid | `(0, −144, 54)` | 76 × 84 × 70 |
| Muzzle ellipsoid | `(0, −181, 40)` | 48 × 64 × 32 |
| Tail capsule | `(0, 117, 7)` to `(0, 186, 51)` | 24 mm diameter |
| Foreleg first pivots | `(±32, −85, −34)` | 64 mm side-to-side spacing |
| Hind-leg first pivots | `(±29, 96, −34)` | 58 mm side-to-side spacing |

Each leg has two transverse M3 pin joints parallel to X. At neutral the second pivot lies 48 mm below the first. The padded upper-link diameter is 26 mm, and the lower link is 20 mm. Centre a 30 × 40 × 20 mm padded foot 61 mm below and 4 mm toward the muzzle from the second pivot. The foot sole is 71 mm below the second pivot. Use mechanical stops for −24° to +39° at the first pivot and −23° to +32° at the lower pivot. Apply a 50 mm lever to each joint at 30° deflection and adjust its return spring to about 10 N at the first pivot or 7 N at the lower one. All eight joints return toward the standing neutral under light displacement.

## Harness and reset

Use a small step-in harness with two distinct leg openings and a 24 mm acetal side-release clasp with two resilient tabs. The clasp has a centre guide between its tabs. Use 12–13 mm wide nylon webbing. The centre bridge is 80 × 13 mm. The four free webbing runs go from the front and rear bridge ends to the two respective buckle strap bars. Set each run to about 150 mm between the stitched attachments. In the laid-flat reset, the two pairs of strap arms start 78 mm apart at the bridge and meet at the clasp ends. Their slack bows gently away from the centreline. Both sides must lie on the table under their own weight.

Put the trunk centre over the table centreline, 285 mm forward of the robot-base baseline. The mannequin stands on its four feet with the trunk reference around 153 mm above the table. Put the bridge on the centreline, 474 mm forward of the base baseline. Point both separated buckle halves inward. Their centres start about 131 mm to either side of the bridge. The setup photo shows the forepaws, leg openings and two buckle halves for repeatable placement.

Perform the buckle proof after the robot's released hold. Pull the two strap bars away from each other along the clasp axis with a spring gauge. Rise gently to 5 N and hold it for one second. Check both tabs remain seated and both forelegs stay threaded. The simulation applies the equivalent pull to a copy of the final released state.
