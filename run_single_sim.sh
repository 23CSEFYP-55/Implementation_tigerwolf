#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_CMD="$VENV_DIR/bin/python"

# Ensure OpenJDK is in PATH if installed via Homebrew
for java_bin in \
    "/opt/homebrew/opt/openjdk@21/bin" \
    "/opt/homebrew/opt/openjdk/bin" \
    "/usr/local/opt/openjdk@21/bin" \
    "/usr/local/opt/openjdk/bin"; do
    if [ -d "$java_bin" ] && [[ ":$PATH:" != *":$java_bin:"* ]]; then
        export PATH="$java_bin:$PATH"
    fi
done

SCHED="${1:-DYNAMIC}"
TRAJ="${2:-PPO}"
TRAJ_UPPER=$(echo "$TRAJ" | tr '[:lower:]' '[:upper:]')
SCHED_UPPER=$(echo "$SCHED" | tr '[:lower:]' '[:upper:]')

# Determine server and port by trajectory
case "$TRAJ_UPPER" in
    CAR)
        TRAJ_SERVER="$PROJECT_DIR/ppo_agent/car_ppo_server.py"
        TRAJ_PORT=5510
        ;;
    TACC|TACC_AAV)
        TRAJ_SERVER="$PROJECT_DIR/ppo_agent/tacc_aav_server.py"
        TRAJ_PORT=5530
        ;;
    *)
        TRAJ_SERVER="$PROJECT_DIR/ppo_agent/tf_ppo_server.py"
        TRAJ_PORT=5500
        ;;
esac

MARL_SERVER="$PROJECT_DIR/ppo_agent/marl_allocation_server.py"
MARL_PORT=5501

free_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        local pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            kill -9 $pids 2>/dev/null || true
        fi
    fi
}

free_port "$TRAJ_PORT"
if [ "$SCHED_UPPER" = "MARL" ]; then
    free_port "$MARL_PORT"
fi

cleanup() {
    if [ -n "$TRAJ_PID" ]; then
        kill "$TRAJ_PID" 2>/dev/null || true
    fi
    if [ -n "$MARL_PID" ]; then
        kill "$MARL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cd "$PROJECT_DIR/ppo_agent"
$PYTHON_CMD "$TRAJ_SERVER" > /dev/null 2>&1 &
TRAJ_PID=$!

if [ "$SCHED_UPPER" = "MARL" ]; then
    $PYTHON_CMD "$MARL_SERVER" > /dev/null 2>&1 &
    MARL_PID=$!
fi

# Wait for ports to open
for i in {1..25}; do
    traj_ready=0
    marl_ready=0
    if nc -z localhost "$TRAJ_PORT" 2>/dev/null; then
        traj_ready=1
    fi
    if [ "$SCHED_UPPER" = "MARL" ]; then
        if nc -z localhost "$MARL_PORT" 2>/dev/null; then
            marl_ready=1
        fi
    else
        marl_ready=1
    fi

    if [ "$traj_ready" -eq 1 ] && [ "$marl_ready" -eq 1 ]; then
        break
    fi
    sleep 0.4
done

cd "$PROJECT_DIR"
java -cp "out/production/iFogSim:jars/*:jars/commons-math3-3.5/*" org.fog.test.perfeval.FANETSimulation "$SCHED" "$TRAJ"
