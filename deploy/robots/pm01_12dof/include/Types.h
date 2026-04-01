#pragma once

// PM01 uses its own SDK communication types.
// TODO: Replace with EngineAI PM01 SDK types when available.
// For now, placeholder using a generic DDS interface.
//
// When the PM01 SDK is available, update these typedefs:
//   using LowCmd_t = engineai::robot::pm01::publisher::LowCmd;
//   using LowState_t = engineai::robot::pm01::subscription::LowState;
//
// The PM01 has 12 actuated DOFs (legs only):
//   Motor indices 0-5:  Left leg  (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
//   Motor indices 6-11: Right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)

#error "PM01 SDK types not yet defined. Replace this header with actual PM01 SDK includes."
