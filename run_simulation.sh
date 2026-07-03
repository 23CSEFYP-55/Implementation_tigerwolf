#!/bin/bash

# Exit immediately if any command fails
set -e

# Define paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_SERVER="$PROJECT_DIR/ppo_agent/fanet_ppo_server.py"

# Function to clean up background processes on exit
cleanup() {
    if [ -n "$PYTHON_PID" ]; then
        kill "$PYTHON_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# 1. Compile Java files
echo "========================================="
echo "Compiling Java simulation files..."
echo "========================================="
javac -d "$PROJECT_DIR/out/production/iFogSim" -cp "$PROJECT_DIR/jars/*:$PROJECT_DIR/jars/commons-math3-3.5/*" $(find "$PROJECT_DIR/src" -name "*.java")
echo "Compilation complete!"

# 2. Check/Start Python Server
if [ -d "$VENV_DIR" ]; then
    PYTHON_CMD="$VENV_DIR/bin/python"
else
    echo "Virtual environment not found at $VENV_DIR."
    echo "Please run setup first or install requirements."
    exit 1
fi

# Install dependencies

echo "========================================="
echo "Installing dependencies..."
echo "========================================="
$PYTHON_CMD -m pip install -r "$PROJECT_DIR/requirements.txt"
echo "Dependencies installed successfully!"

echo "========================================="
echo "Starting Python PPO Server..."
echo "========================================="
cd "$PROJECT_DIR/ppo_agent"
$PYTHON_CMD "$PYTHON_SERVER" &
PYTHON_PID=$!

echo "Waiting for Python PPO Server to start listening on port 5500..."
# Wait for port 5500 to be open
for i in {1..20}; do
    if nc -z localhost 5500 2>/dev/null; then
        break
    fi
    sleep 0.5
done

if ! nc -z localhost 5500 2>/dev/null; then
    echo "Error: Python PPO Server failed to start."
    exit 1
fi
echo "Python PPO Server is ready!"

# 3. Start Java Simulation
echo "========================================="
echo "Starting Java Simulation (iFogSim)..."
echo "========================================="
cd "$PROJECT_DIR"
java -cp "out/production/iFogSim:jars/*:jars/commons-math3-3.5/*" org.fog.test.perfeval.FANETSimulation

echo "Simulation finished successfully!"
