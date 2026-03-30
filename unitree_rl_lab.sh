#!/usr/bin/env bash

export UNITREE_RL_LAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if ! [[ -z "${VIRTUAL_ENV}" ]]; then
    python_exe=${VIRTUAL_ENV}/bin/python
else
    echo "[Error] No virtual environment activated. Please activate the virtual environment first."
    # exit 1
fi


# task env name autocomplete
_ut_rl_lab_python_argcomplete_wrapper() {
    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}
complete -o nospace -F _ut_rl_lab_python_argcomplete_wrapper "./unitree_rl_lab.sh"


_ut_setup_venv() {

    # add source unitree_rl_lab.sh to venv activate script
    local activate_script=${VIRTUAL_ENV}/bin/activate

    # append environment setup to the activate script if not already present
    if ! grep -q "# for unitree_rl_lab" "${activate_script}" 2>/dev/null; then
        printf '\n%s\n' \
            '# for Isaac Lab' \
            'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
            'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
            '' \
            '# show icon if not running headless' \
            'export RESOURCE_NAME="IsaacSim"' \
            '' \
            '# for unitree_rl_lab' \
            'source '${UNITREE_RL_LAB_PATH}'/unitree_rl_lab.sh' \
            '' >> "${activate_script}"
    fi

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    # we need to setup variables to load the binaries
    local isaacsim_setup_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_env_script}" ]; then
        if ! grep -q "# for Isaac Sim" "${activate_script}" 2>/dev/null; then
            printf '%s\n' \
                '# for Isaac Sim' \
                'source '${isaacsim_setup_env_script}'' \
                '' >> "${activate_script}"
        fi
    fi
}

# pass the arguments
case "$1" in
    -i|--install)
        git lfs install # ensure git lfs is installed
        pip install -e ${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab/
        _ut_setup_venv
        activate-global-python-argcomplete
        ;;
    -l|--list)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/list_envs.py "$@"
        ;;
    -p|--play)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/play.py "$@"
        ;;
    -t|--train)
        shift
        ${python_exe} ${UNITREE_RL_LAB_PATH}/scripts/rsl_rl/train.py --headless "$@"
        ;;
    *) # unknown option
        ;;
esac
