# PM01 12-DOF Deployment

## Status: TEMPLATE — NOT READY FOR HARDWARE

This deploy configuration is adapted from the G1 29-DOF controller. It requires the following before use on real hardware:

### TODO

1. **PM01 SDK Integration**
   - Replace `Types.h` with actual EngineAI PM01 SDK includes
   - Update `main.cpp` DDS initialization for PM01
   - Verify motor indexing matches `joint_ids_map` in deploy.yaml

2. **Export ONNX Model**
   From training logs, export the best checkpoint:
   ```bash
   # TODO: Add export script or use IsaacLab's built-in export
   ```
   Place the exported model at: `config/policy/velocity/v0/exported/policy.onnx`

3. **Verify PD Gains**
   The stiffness/damping values in `deploy.yaml` are from the simulation config.
   These may need tuning for the real robot (sim-to-real gap).

4. **Verify Joint Mapping**
   `joint_ids_map: [0,1,2,3,4,5,6,7,8,9,10,11]` assumes PM01 SDK motor indices
   match the policy output order. Verify against PM01 hardware documentation.

## Directory Structure

```
pm01_12dof/
├── CMakeLists.txt                         # Build config
├── main.cpp                               # Entry point
├── README.md                              # This file
├── config/
│   ├── config.yaml                        # FSM configuration
│   └── policy/velocity/v0/
│       ├── exported/policy.onnx           # TODO: Export from training
│       └── params/deploy.yaml             # Deploy parameters
├── include/
│   └── Types.h                            # TODO: PM01 SDK types
└── src/
    └── State_RLBase.cpp                   # RL velocity control state
```

## Joint Order (Policy Output)

| Index | Joint Name | SDK Motor |
|-------|-----------|-----------|
| 0 | j00_hip_pitch_l | 0 |
| 1 | j01_hip_roll_l | 1 |
| 2 | j02_hip_yaw_l | 2 |
| 3 | j03_knee_pitch_l | 3 |
| 4 | j04_ankle_pitch_l | 4 |
| 5 | j05_ankle_roll_l | 5 |
| 6 | j06_hip_pitch_r | 6 |
| 7 | j07_hip_roll_r | 7 |
| 8 | j08_hip_yaw_r | 8 |
| 9 | j09_knee_pitch_r | 9 |
| 10 | j10_ankle_pitch_r | 10 |
| 11 | j11_ankle_roll_r | 11 |

## Build

```bash
cd deploy/robots/pm01_12dof
mkdir build && cd build
cmake ..
make
```

## Run

```bash
./pm01_ctrl --network <network_interface>
```

Controls:
- `L2 + Up` — Enter FixStand mode
- `R1 + X` — Start velocity control (RL policy)
- `L2 + B` — Return to Passive mode
- `W/A/S/D/Q/E` — Velocity commands (keyboard)
