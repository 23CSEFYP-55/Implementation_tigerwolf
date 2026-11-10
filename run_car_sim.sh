#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
CAR_SERVER="$PROJECT_DIR/ppo_agent/car_ppo_server.py"
MARL_SERVER="$PROJECT_DIR/ppo_agent/marl_allocation_server.py"

cleanup() {
    if [ -n "$CAR_PID" ]; then
        kill "$CAR_PID" 2>/dev/null || true
    fi
    if [ -n "$MARL_PID" ]; then
        kill "$MARL_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

PYTHON_CMD="$VENV_DIR/bin/python"

cd "$PROJECT_DIR/ppo_agent"
$PYTHON_CMD "$CAR_SERVER" > /dev/null 2>&1 &
CAR_PID=$!

$PYTHON_CMD "$MARL_SERVER" > /dev/null 2>&1 &
MARL_PID=$!

for i in {1..60}; do
    if nc -z localhost 5510 2>/dev/null && nc -z localhost 5501 2>/dev/null; then
        break
    fi
    sleep 0.5
done

cd "$PROJECT_DIR"
java -cp "out/production/iFogSim:jars/*:jars/commons-math3-3.5/*" org.fog.test.perfeval.FANETSimulation ${1:-DYNAMIC} CAR