package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.*;

/**
 * APRepairEngine (Adaptive-Predictive Dynamic Repair Engine)
 *
 * Implements:
 * 1. Urgency / Slack-Prioritized Min-Heap Repair Queue:
 *    Orders incoming and repaired tasks by normalized deadline slack S_k.
 * 2. Proactive Swarm Load Balancing:
 *    Direct assignment soft-cap (25 tasks/batch) that triggers cooperative P2P offloading
 *    to nearby underloaded peers (d <= 500m), before falling back to full hard buffer capacity (150).
 * 3. Multi-Candidate Feasibility Search:
 *    Ranks all in-range UAVs using multi-objective utility (link proximity, CPU MIPS, active queue load).
 * 4. Pareto-Optimal Cooperative P2P Offloading:
 *    Identifies optimal peer UAVs within 500m of busy edge nodes, balancing MIPS,
 *    available queue headroom, and mesh relay distance.
 */
public class APRepairEngine {

    private final AssignmentState assignmentState;
    private static final double DATA_RATE = 6250000.0; // 50 Mbps in bytes/sec
    private static final int SOFT_LOAD_CAP = 12;       // Soft threshold triggering cooperative P2P sharing

    public APRepairEngine(AssignmentState state) {
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

        Set<Integer> uavsToInvalidate = new HashSet<>();
        for (Integer uavId : assignmentState.getAllAssignments().keySet()) {
            if (!activeUavIds.contains(uavId)) {
                uavsToInvalidate.add(uavId);
            }
        }

        for (Integer uavId : uavsToInvalidate) {
            invalidTasks.addAll(assignmentState.invalidateAssignments(uavId));
        }

        return invalidTasks;
    }

    /**
     * Assigns tasks using a Slack-Prioritized Min-Heap queue and multi-criteria utility.
     */
    public long assignNewTasks(List<Tuple> taskPool, List<FogDevice> uavs) {
        long reassignedCount = 0;
        if (taskPool == null || taskPool.isEmpty() || uavs == null || uavs.isEmpty()) {
            return 0;
        }

        Map<Integer, FogDevice> uavMap = new HashMap<>();
        Map<Integer, Integer> batchAssignedCount = new HashMap<>();
        Map<Integer, Double> batchMiInQueue = new HashMap<>();

        for (FogDevice uav : uavs) {
            uavMap.put(uav.getId(), uav);
            batchAssignedCount.put(uav.getId(), 0);
            batchMiInQueue.put(uav.getId(), 0.0);
        }

        // 1. Construct Slack-Prioritized Min-Heap: tightest deadline slack processed first
        PriorityQueue<Tuple> priorityRepairQueue = new PriorityQueue<>((t1, t2) -> {
            double slack1 = estimateInitialSlack(t1);
            double slack2 = estimateInitialSlack(t2);
            return Double.compare(slack1, slack2);
        });

        for (Tuple task : taskPool) {
            priorityRepairQueue.add(task);
        }

        // 2. Incremental Assignment Loop
        while (!priorityRepairQueue.isEmpty()) {
            Tuple task = priorityRepairQueue.poll();
            if (task.assignedUavId != null) {
                reassignedCount++;
            }
            task.assignedUavId = null;

            // Phase A: Try direct assignment under soft load cap (proactive fairness)
            FogDevice bestUav = findBestFeasibleUAV(task, uavs, batchAssignedCount, batchMiInQueue, SOFT_LOAD_CAP);

            if (bestUav != null) {
                bindTaskToUav(task, bestUav, batchAssignedCount, batchMiInQueue);
            } else {
                // Phase B: Proactive Pareto-optimal cooperative P2P offloading to underloaded peers
                boolean offloaded = attemptParetoPeerOffload(task, uavs, batchAssignedCount, batchMiInQueue, SOFT_LOAD_CAP);

                if (!offloaded) {
                    // Phase C: If peers are also busy, fallback to direct assignment up to hard capacity (150)
                    FogDevice fallbackUav = findBestFeasibleUAV(task, uavs, batchAssignedCount, batchMiInQueue, Integer.MAX_VALUE);
                    if (fallbackUav != null) {
                        bindTaskToUav(task, fallbackUav, batchAssignedCount, batchMiInQueue);
                    } else {
                        // Phase D: Deep P2P fallback up to hard capacity
                        attemptParetoPeerOffload(task, uavs, batchAssignedCount, batchMiInQueue, Integer.MAX_VALUE);
                    }
                }
            }
        }

        return reassignedCount;
    }

    private void bindTaskToUav(Tuple task, FogDevice uav,
                               Map<Integer, Integer> batchAssignedCount,
                               Map<Integer, Double> batchMiInQueue) {
        int uId = uav.getId();
        uav.acceptedTasks.add(task);
        assignmentState.addTask(uId, task);
        task.assignedUavId = uId;
        GlobalState.accumulatedThroughput += task.tupleDataSize;

        batchAssignedCount.put(uId, batchAssignedCount.get(uId) + 1);
        batchMiInQueue.put(uId, batchMiInQueue.get(uId) + task.getCloudletLength());
    }

    /**
     * Evaluates all candidate UAVs covering the MD, sorting them by multi-objective utility.
     * Iterates through candidate UAVs until a feasible one (capacity + latency) is found.
     */
    private FogDevice findBestFeasibleUAV(Tuple task, List<FogDevice> uavs,
                                         Map<Integer, Integer> batchAssignedCount,
                                         Map<Integer, Double> batchMiInQueue,
                                         int maxAllowedLoad) {
        FogDevice md = getMdForTask(task);
        if (md == null) return null;

        List<UavCandidate> candidates = new ArrayList<>();

        for (FogDevice uav : uavs) {
            double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));

            if (dist <= uav.coverageRadius) {
                double r_up = 1.0 / (1.0 + dist);
                double r_down = 1.0 / (1.0 + dist);
                int currentLoad = batchAssignedCount.getOrDefault(uav.getId(), 0);
                double loadPenalty = 1.0 / (1.0 + currentLoad);
                double mipsRatio = uav.getHost().getTotalMips() / 26000.0;

                // Multi-objective utility: balances proximity, active batch load, and CPU computing capacity
                double utility = 0.30 * r_up + 0.30 * r_down + 0.20 * loadPenalty + 0.20 * mipsRatio;
                candidates.add(new UavCandidate(uav, utility));
            }
        }

        // Sort descending by utility
        candidates.sort((c1, c2) -> Double.compare(c2.utility, c1.utility));

        // Assign to the highest utility UAV that satisfies queue capacity and deadline latency
        for (UavCandidate cand : candidates) {
            FogDevice u = cand.uav;
            int currentLoad = batchAssignedCount.getOrDefault(u.getId(), 0);
            int effectiveCap = Math.min(u.taskCapacity, maxAllowedLoad);
            if (currentLoad < effectiveCap && checkBatchLatencyConstraint(u, task, batchMiInQueue.getOrDefault(u.getId(), 0.0))) {
                return u;
            }
        }

        return null;
    }

    /**
     * Pareto-optimal cooperative P2P offloading: finds the closest bridge UAV covering the MD,
     * then searches all peer UAVs within 500m to offload the task.
     */
    private boolean attemptParetoPeerOffload(Tuple task, List<FogDevice> uavs,
                                            Map<Integer, Integer> batchAssignedCount,
                                            Map<Integer, Double> batchMiInQueue,
                                            int maxAllowedLoad) {
        FogDevice md = getMdForTask(task);
        if (md == null) return false;

        // Find the bridge UAV covering the MD
        FogDevice bridgeUav = null;
        double minBridgeDist = Double.MAX_VALUE;
        for (FogDevice uav : uavs) {
            double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));
            if (dist <= uav.coverageRadius && dist < minBridgeDist) {
                minBridgeDist = dist;
                bridgeUav = uav;
            }
        }

        if (bridgeUav == null) return false;

        // Search for Pareto-optimal peer among 1-hop UAV mesh neighbors (d <= 500m)
        FogDevice bestPeer = null;
        double bestPeerScore = -Double.MAX_VALUE;

        for (FogDevice peer : uavs) {
            if (peer.getId() == bridgeUav.getId()) continue;

            double distToPeer = Math.sqrt(Math.pow(bridgeUav.x_coord - peer.x_coord, 2) +
                    Math.pow(bridgeUav.y_coord - peer.y_coord, 2));

            if (distToPeer <= 500.0) {
                int currentLoad = batchAssignedCount.getOrDefault(peer.getId(), 0);
                int effectiveCap = Math.min(peer.taskCapacity, maxAllowedLoad);
                if (currentLoad < effectiveCap && checkBatchLatencyConstraint(peer, task, batchMiInQueue.getOrDefault(peer.getId(), 0.0))) {
                    double mipsScore = peer.getHost().getTotalMips() / 26000.0;
                    double capScore = 1.0 - ((double) currentLoad / (double) Math.max(1, peer.taskCapacity));
                    double distPenalty = distToPeer / 500.0;

                    double peerScore = 0.45 * mipsScore + 0.40 * capScore - 0.15 * distPenalty;

                    if (peerScore > bestPeerScore) {
                        bestPeerScore = peerScore;
                        bestPeer = peer;
                    }
                }
            }
        }

        if (bestPeer != null) {
            int pId = bestPeer.getId();
            bestPeer.acceptedTasks.add(task);
            assignmentState.addTask(pId, task);
            task.assignedUavId = pId;
            GlobalState.accumulatedThroughput += task.tupleDataSize;
            bridgeUav.tasksOffloaded++;

            batchAssignedCount.put(pId, batchAssignedCount.get(pId) + 1);
            batchMiInQueue.put(pId, batchMiInQueue.get(pId) + task.getCloudletLength());
            return true;
        }

        return false;
    }

    private double estimateInitialSlack(Tuple task) {
        double txDelayMs = (task.getCloudletFileSize() / DATA_RATE) * 1000.0;
        double avgMips = 14000.0; // Fleet average MIPS
        double procDelayMs = (task.getCloudletLength() / avgMips) * 1000.0;
        double estimatedTotal = txDelayMs + procDelayMs;
        if (task.tolerantLatency <= 0) return 0.0;
        return (task.tolerantLatency - estimatedTotal) / task.tolerantLatency;
    }

    private boolean checkBatchLatencyConstraint(FogDevice uav, Tuple task, double currentBatchMiInQueue) {
        double txDelayMs = (task.getCloudletFileSize() / DATA_RATE) * 1000.0;
        double uavMips = uav.getHost().getTotalMips();

        double predictedQueueDelayMs = ((currentBatchMiInQueue + task.getCloudletLength()) / uavMips) * 1000.0;
        double totalDelay = txDelayMs + predictedQueueDelayMs;

        return totalDelay <= task.tolerantLatency;
    }

    private FogDevice getMdForTask(Tuple task) {
        try {
            int srcId = task.getSourceDeviceId();
            return (FogDevice) org.cloudbus.cloudsim.core.CloudSim.getEntity(srcId);
        } catch (Exception e) {
            return null;
        }
    }

    private static class UavCandidate {
        final FogDevice uav;
        final double utility;

        UavCandidate(FogDevice uav, double utility) {
            this.uav = uav;
            this.utility = utility;
        }
    }
}
