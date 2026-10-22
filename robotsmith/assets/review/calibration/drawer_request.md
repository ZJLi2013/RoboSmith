Build this item and follow CONTRACT.md:

A tabletop drawer cabinet for a robot arm to open, put a small object in, and close.
- Overall about 0.42 m wide (Y) x 0.30 m deep (X) x 0.25 m tall (Z), short legs, one drawer,
  the drawer front faces +X.
- Names: item `drawer_cabinet`; parts `carcass` (root), `drawer`, `handle`; prismatic joint
  `drawer_slide` along +X from closed (0) to about 0.24 m open, with the drawer still held by
  the carcass at full travel; place target `drawer_opening` on the drawer floor object
  `tray_floor`.
- The drawer is an open tray (floor, back, two sides, front panel), its inside at least
  0.25 x 0.35 m and 0.08 m deep, clear of the carcass when it slides.
- The handle is a separate part attached to the drawer: a thin neck object `handle_neck`
  (at most 15 mm across, sticking out at least 30 mm from the drawer front, centred on it)
  ending in a knob object `handle_knob` about 35 mm across. A gripper closes on the neck
  from above and pulls against the knob.
- The carcass is open at the front where the drawer sits; leave a few millimetres of gap
  between the drawer and the carcass on every side.
