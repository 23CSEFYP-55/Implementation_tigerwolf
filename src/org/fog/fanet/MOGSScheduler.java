package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.*;

public class MOGSScheduler implements TaskScheduler {

    @Override
    public void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs) {
        Queue<Tuple> unassignedTasks = new LinkedList<>(tasks);

        // 1. Active Party: MDs Rank UAVs
        for (Tuple task : tasks) {
            task.preferredUAVs = new ArrayList<>();
            Map<Integer, Double> uavScores = new HashMap<>();
            
            FogDevice md = getMdForTask(task);
            if (md == null) continue;
            
            for (FogDevice uav : uavs) {
                double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));
                if (dist <= uav.coverageRadius) {
                    double r_up = 1.0 / (1.0 + dist);
                    double r_down = 1.0 / (1.0 + dist);
                    double load_penalty = 1.0 / (1.0 + uav.acceptedTasks.size());
                    
                    double alpha = 0.4, beta = 0.4, gamma = 0.2;
                    double score = alpha * r_up + beta * r_down + gamma * load_penalty;
                    
                    uavScores.put(uav.getId(), score);
                }
            }
            
            if (uavScores.isEmpty()) {
                // System.out.println("MOGS DEBUG: Task " + task.getCloudletId() + " has no UAVs in range. MD: " + md.getName() + " coords: " + md.x_coord + "," + md.y_coord);
            }
            
            List<Integer> sortedUavs = new ArrayList<>(uavScores.keySet());
            sortedUavs.sort((id1, id2) -> Double.compare(uavScores.get(id2), uavScores.get(id1)));
            task.preferredUAVs = sortedUavs;
            task.currentProposalIndex = 0;
        }

        // 2. MOGS Matching Process
        Map<Integer, FogDevice> uavMap = new HashMap<>();
        for (FogDevice uav : uavs) uavMap.put(uav.getId(), uav);

        while (!unassignedTasks.isEmpty()) {
            Tuple task = unassignedTasks.poll();

            if (task.currentProposalIndex >= task.preferredUAVs.size()) {
                task.assignedUavId = null; // Rejected by all in range
                continue;
            }

            int targetUavId = task.preferredUAVs.get(task.currentProposalIndex);
            FogDevice targetUav = uavMap.get(targetUavId);
            task.currentProposalIndex++;

            // Passive Party: UAV Accepts based on Latency and Queue
            // High/Low load state check
            if (targetUav.acceptedTasks.size() < targetUav.taskCapacity && checkLatencyConstraint(targetUav, task)) {
                targetUav.acceptedTasks.add(task);
                GlobalState.accumulatedThroughput += task.tupleDataSize;
            } else {
                // P2P Offloading logic: Try to offload to a low-load peer
                boolean offloaded = false;
                for (FogDevice peer : uavs) {
                    if (peer.getId() != targetUav.getId()) {
                        System.out.println("DEBUG: P2P Evaluation - targetUav " + targetUav.getName() + " queue full. Evaluating peer " + peer.getName() + " queue size: " + peer.acceptedTasks.size());
                        if (peer.acceptedTasks.size() < peer.taskCapacity && checkLatencyConstraint(peer, task)) {
                            double distToPeer = Math.sqrt(Math.pow(uavMap.get(targetUavId).x_coord - peer.x_coord, 2) + 
                                                          Math.pow(uavMap.get(targetUavId).y_coord - peer.y_coord, 2));
                            // If within comm range (e.g. 500m)
                            if (distToPeer < 500.0) {
                                peer.acceptedTasks.add(task);
                                task.assignedUavId = peer.getId();
                                GlobalState.accumulatedThroughput += task.tupleDataSize;
                                targetUav.tasksOffloaded++; // Increment offloaded count
                                offloaded = true;
                                break;
                            }
                        }
                    }
                }
                
                if (!offloaded) {
                    // Reject, goes back in queue to try next choice
                    unassignedTasks.add(task);
                }
            }
        }
        
        for (FogDevice uav : uavs) {
            for (Tuple accepted : uav.acceptedTasks) {
                if (accepted.assignedUavId == null) {
                    accepted.assignedUavId = uav.getId();
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
        for (Tuple t : uav.acceptedTasks) totalMiInQueue += t.getCloudletLength();
        
        double predictedQueueDelayMs = ((totalMiInQueue + task.getCloudletLength()) / uavMips) * 1000.0;
        double totalPredictedLatency = transmissionDelayMs + predictedQueueDelayMs;
        
        return totalPredictedLatency <= task.tolerantLatency;
    }
}