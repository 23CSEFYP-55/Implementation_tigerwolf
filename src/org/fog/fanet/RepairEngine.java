package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.*;

/**
 * Handles the logic for detecting invalid assignments and assigning tasks 
 * dynamically without full recomputation.
 */
public class RepairEngine {
    
    private AssignmentState assignmentState;

    public RepairEngine(AssignmentState state) {
        this.assignmentState = state;
    }

    /**
     * Refreshes UAV state, identifying any UAVs that have been removed or are unreachable.
     * Invalidates their assignments and returns those tasks to be repaired.
     */
    public List<Tuple> refreshUAVState(List<FogDevice> activeUavs) {
        List<Tuple> invalidTasks = new ArrayList<>();
        Set<Integer> activeUavIds = new HashSet<>();
        
        for (FogDevice uav : activeUavs) {
            activeUavIds.add(uav.getId());
        }

        // Identify UAVs in the assignment state that are no longer active
        Set<Integer> uavsToInvalidate = new HashSet<>();
        for (Integer uavId : assignmentState.getAllAssignments().keySet()) {
            if (!activeUavIds.contains(uavId)) {
                uavsToInvalidate.add(uavId);
            }
        }

        // Invalidate and collect tasks for repair
        for (Integer uavId : uavsToInvalidate) {
            invalidTasks.addAll(assignmentState.invalidateAssignments(uavId));
        }

        return invalidTasks;
    }

    /**
     * Attempts to assign tasks from the repair queue (invalid + new).
     */
    public void assignNewTasks(Queue<Tuple> repairQueue, List<FogDevice> uavs) {
        Map<Integer, FogDevice> uavMap = new HashMap<>();
        for (FogDevice uav : uavs) {
            uavMap.put(uav.getId(), uav);
        }

        while (!repairQueue.isEmpty()) {
            Tuple task = repairQueue.poll();
            task.assignedUavId = null; // Reset previous assignment if any

            FogDevice bestUav = findBestUAV(task, uavs);
            
            if (bestUav != null) {
                // Assign to best UAV
                bestUav.acceptedTasks.add(task);
                assignmentState.addTask(bestUav.getId(), task);
                task.assignedUavId = bestUav.getId();
                PreferenceBuilder.accumulatedThroughput += task.tupleDataSize;
            } else {
                // Attempt peer-to-peer offloading if direct assignment failed
                attemptPeerOffload(task, uavs, uavMap);
            }
        }
    }

    /**
     * Finds the UAV with the highest utility score for the given task, 
     * ensuring capacity and latency constraints are met.
     */
    private FogDevice findBestUAV(Tuple task, List<FogDevice> uavs) {
        FogDevice md = getMdForTask(task);
        if (md == null) return null;

        FogDevice bestUav = null;
        double maxScore = -Double.MAX_VALUE;

        for (FogDevice uav : uavs) {
            double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));
            
            // Check basic coverage radius
            if (dist <= uav.coverageRadius) {
                // Check feasibility: Capacity + Latency
                if (uav.acceptedTasks.size() < uav.taskCapacity && checkLatencyConstraint(uav, task)) {
                    
                    double r_up = 1.0 / (1.0 + dist);
                    double r_down = 1.0 / (1.0 + dist);
                    double load_penalty = 1.0 / (1.0 + uav.acceptedTasks.size());
                    
                    double alpha = 0.4, beta = 0.4, gamma = 0.2;
                    double score = alpha * r_up + beta * r_down + gamma * load_penalty;
                    
                    if (score > maxScore) {
                        maxScore = score;
                        bestUav = uav;
                    }
                }
            }
        }
        
        return bestUav;
    }

    /**
     * Attempts peer-to-peer offloading by searching for nearby UAVs 
     * that can feasibly compute the task.
     */
    private void attemptPeerOffload(Tuple task, List<FogDevice> uavs, Map<Integer, FogDevice> uavMap) {
        FogDevice md = getMdForTask(task);
        if (md == null) return;
        
        // Find the "preferred" UAV that was in range but full/violating latency, to serve as the offload bridge
        FogDevice bridgeUav = null;
        for (FogDevice uav : uavs) {
            double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));
            if (dist <= uav.coverageRadius) {
                bridgeUav = uav; // Just take the first one found as the bridge
                break;
            }
        }
        
        if (bridgeUav == null) return; // No UAV in range of MD to even initiate an offload
        
        // Search peers of the bridge UAV
        for (FogDevice peer : uavs) {
            if (peer.getId() != bridgeUav.getId()) {
                if (peer.acceptedTasks.size() < peer.taskCapacity && checkLatencyConstraint(peer, task)) {
                    double distToPeer = Math.sqrt(Math.pow(bridgeUav.x_coord - peer.x_coord, 2) + 
                                                  Math.pow(bridgeUav.y_coord - peer.y_coord, 2));
                    // If within comm range (e.g. 500m)
                    if (distToPeer < 500.0) {
                        peer.acceptedTasks.add(task);
                        assignmentState.addTask(peer.getId(), task);
                        task.assignedUavId = peer.getId();
                        PreferenceBuilder.accumulatedThroughput += task.tupleDataSize;
                        bridgeUav.tasksOffloaded++; // Increment offloaded count
                        return; // Offload successful
                    }
                }
            }
        }
    }

    private FogDevice getMdForTask(Tuple task) {
        try {
            int srcId = task.getSourceDeviceId();
            return (FogDevice) org.cloudbus.cloudsim.core.CloudSim.getEntity(srcId);
        } catch (Exception e) {
            return null;
        }
    }

    private boolean checkLatencyConstraint(FogDevice uav, Tuple task) {
        double currentDataRate = 6250000.0; // 50 Mbps in bytes/sec
        double transmissionDelayMs = (task.getCloudletFileSize() / currentDataRate) * 1000.0;
        
        double uavMips = uav.getHost().getTotalMips();
        double totalMiInQueue = 0.0;
        for (Tuple t : uav.acceptedTasks) {
            totalMiInQueue += t.getCloudletLength();
        }
        
        double predictedQueueDelayMs = ((totalMiInQueue + task.getCloudletLength()) / uavMips) * 1000.0;
        double totalPredictedLatency = transmissionDelayMs + predictedQueueDelayMs;
        
        return totalPredictedLatency <= task.tolerantLatency;
    }
}
