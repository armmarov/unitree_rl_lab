#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = std::make_shared<Keyboard>();

void init_fsm_state()
{
    // TODO: Replace with PM01 SDK initialization
    // auto lowcmd_sub = std::make_shared<engineai::robot::pm01::subscription::LowCmd>();
    // usleep(0.2 * 1e6);
    // if(!lowcmd_sub->isTimeout())
    // {
    //     spdlog::critical("The other process is using the lowcmd channel, please close it first.");
    //     exit(0);
    // }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- EngineAI Robotics --- \n";
    std::cout << "     PM01-12dof Controller \n";

    // TODO: Replace with PM01 SDK DDS config
    // engineai::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // PM01 is 12-DOF (legs only), no mode_machine check needed
    // unless PM01 SDK has a similar mechanism

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";

    while (true)
    {
        sleep(1);
    }

    return 0;
}
