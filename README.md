# iFogSim FANET Simulation with PPO Agent

This repository contains an iFogSim-based simulation of a Flying Ad-Hoc Network (FANET), integrated with a Python-based Proximal Policy Optimization (PPO) agent for intelligent task offloading and resource scheduling.

## Architecture & Workflow

The project consists of two main components acting in tandem:
1. **Java Simulation (iFogSim)**: Simulates the FANET environment, including UAV fog nodes, task generation, network latency, queueing state, and computational energy consumption.
2. **Python PPO Server**: A standalone Gym environment and Stable-Baselines3 server that listens for integration requests from the Java simulation. It tracks real-time exact queue states and coordinates of the UAV nodes and responds with the optimal UAV assignment based on a physics-based ranking score AI tie-breaker.

## Requirements & Setup

### Java (Simulation Environment)
- Java 8 (JRE 1.8)
- External libraries inside the `jars` folder. If importing into IntelliJ IDEA or Eclipse, ensure the JARs are added to the project's build path/library dependency list.

### Python (PPO Server)
- Python 3.8 or higher is recommended.
- Install the required Machine Learning and environment dependencies:
```bash
pip install -r requirements.txt
```

## Running the Simulation

**Step 1. Start the PPO Server**
Navigate into the `ppo_agent` directory and start the Python server.
```bash
cd ppo_agent
python fanet_ppo_server.py
```
The server will boot up and start listening on port `5000`. It will attempt to load an existing pre-trained model (`fanet_ppo_model.zip`), or train a new model if one is not found.

**Step 2. Start the Java Simulation**
Run the main FANET simulation class located at:
`src/org/fog/test/perfeval/FANETSimulation.java`
Upon starting, the Java simulation connects to `localhost:5000`. The continuous task queuing values stream to the Python server, which calculates and returns the optimal offloading assignment back to the iFogSim broker.


---

### Legacy iFogSim2 README
 A Toolkit for Modeling and Simulation of Resource Management Techniques in Internet of Things, Edge and Fog Computing Environments with the following new features:
 * Mobility-support and Migration Management
   * Supporting real mobility datasets
   * Implementing different random mobility models 
 * Microservice Orchestration
 * Dynamic Distributed Clustering
 * Any Combinations of Above-mentioned Features
 * Full Compatibility with the Latest Version of the CloudSim (i.e., (https://github.com/Cloudslab/cloudsim/releases)) and [Previous iFogSim Version](https://github.com/Cloudslab/iFogSim1) and Tutorials

iFogSim2 currently encompasses several new usecases such as:
 * Audio Translation Scenario
 * Healthcare Scenario
 * Crowd-sensing Scenario

# References
 * Redowan Mahmud, Samodha Pallewatta, Mohammad Goudarzi, and Rajkumar Buyya, <A href="https://arxiv.org/abs/2109.05636">iFogSim2: An Extended iFogSim Simulator for Mobility, Clustering, and Microservice Management in Edge and Fog Computing Environments</A>, Journal of Systems and Software (JSS), Volume 190, Pages: 1-17, ISSN:0164-1212, Elsevier Press, Amsterdam, The Netherlands, August 2022.
 * Harshit Gupta, Amir Vahid Dastjerdi , Soumya K. Ghosh, and Rajkumar Buyya, <A href="http://www.buyya.com/papers/iFogSim.pdf">iFogSim: A Toolkit for Modeling and Simulation of Resource Management Techniques in Internet of Things, Edge and Fog Computing Environments</A>, Software: Practice and Experience (SPE), Volume 47, Issue 9, Pages: 1275-1296, ISSN: 0038-0644, Wiley Press, New York, USA, September 2017.
 * Redowan Mahmud and Rajkumar Buyya, <A href="http://www.buyya.com/papers/iFogSim-Tut.pdf">Modelling and Simulation of Fog and Edge Computing Environments using iFogSim Toolkit</A>, Fog and Edge Computing: Principles and Paradigms, R. Buyya and S. Srirama (eds), 433-466pp, ISBN: 978-111-95-2498-4, Wiley Press, New York, USA, January 2019.
