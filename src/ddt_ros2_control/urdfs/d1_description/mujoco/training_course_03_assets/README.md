# D1 training course 03

Generated from `tools/course03_layout.json` by `tools/generate_training_course.py`.
The scene includes the original `robot.xml`; the generator does not alter source
assets, contact friction, robot parameters or weights. The manifest records SHA256
hashes of the included robot, its meshes and the original scene where present.

The center lane is 20 m of open level ground. South bays contain independent
5/15/30/50 mm steps, 15/30/60 mm riser staircases, fixed low broken plates and
slalom posts. North bays contain 3/6/10 degree continuous ramps, 10/25/50 mm
colliding heightfields and gates with 1.5/1.1/0.75 m clear openings. Floor guides
and entry marks are visual only (`contype=0`, `conaffinity=0`); obstacles collide.
The original `(0, 0, 0.8)` spawn is unobstructed. Open ground connects every bay.

The scene is for staged training and manual inspection. No controller success or
traversal has been verified. Colors show intended relative difficulty, not a
measured capability threshold. Geometry uses solid boxes/cylinders and PNG hfields.
MuJoCo resolves hfield paths relative to the robot compiler's `meshdir`, hence
the generated `../mujoco/training_course_03_assets/...` references.

Regeneration requires a new, empty destination: the generator refuses to replace
the scene or asset directory. It does not start a viewer or step a simulation.
