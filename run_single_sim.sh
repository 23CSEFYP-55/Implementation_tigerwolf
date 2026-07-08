#!/bin/bash
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_SERVER="$PROJECT_DIR/ppo_agent/fanet_ppo_server.py"

cleanup() {
    if [ -n "$PYTHON_PID" ]; then
        kill "$PYTHON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

PYTHON_CMD="$VENV_DIR/bin/python"

cd "$PROJECT_DIR/ppo_agent"
$PYTHON_CMD "$PYTHON_SERVER" > /dev/null 2>&1 &
PYTHON_PID=$!

for i in {1..20}; do
    if nc -z localhost 5500 2>/dev/null; then
        break
    fi
    sleep 0.5
done

cd "$PROJECT_DIR"
java -cp "out/production/iFogSim:jars/*:jars/commons-math3-3.5/*" org.fog.test.perfeval.FANETSimulation
