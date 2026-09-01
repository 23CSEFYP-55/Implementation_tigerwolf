package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.*;

public class BiddingScheduler implements TaskScheduler {

    private double phi = 0.95;
    private double beta = 0.01;
    private double v = 5.0;
    private double delta = 1.0;

    @Override
    public void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs) {
        if (tasks.isEmpty() || uavs.isEmpty()) return;

        Map<Integer, FogDevice> uavMap = new HashMap<>();
        for (FogDevice uav : uavs) uavMap.put(uav.getId(), uav);

        Map<Integer, Tuple> taskMap = new HashMap<>();
        for (Tuple t : tasks) taskMap.put(t.getCloudletId(), t);

        // Precompute distances
        Map<Integer, Map<Integer, Double>> d_ut = new HashMap<>(); // uavId -> taskId -> dist
        for (FogDevice uav : uavs) {
            d_ut.put(uav.getId(), new HashMap<>());
            for (Tuple task : tasks) {
                FogDevice md = getMdForTask(task);
                if (md != null) {
                    double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));
                    d_ut.get(uav.getId()).put(task.getCloudletId(), dist);
                }
            }
        }

        Map<Integer, Map<Integer, Double>> d_tt = new HashMap<>(); // taskId -> taskId -> dist
        for (Tuple t1 : tasks) {
            d_tt.put(t1.getCloudletId(), new HashMap<>());
            FogDevice md1 = getMdForTask(t1);
            for (Tuple t2 : tasks) {
                FogDevice md2 = getMdForTask(t2);
                if (md1 != null && md2 != null) {
                    double dist = Math.sqrt(Math.pow(md1.x_coord - md2.x_coord, 2) + Math.pow(md1.y_coord - md2.y_coord, 2));
                    d_tt.get(t1.getCloudletId()).put(t2.getCloudletId(), dist);
                }
            }
        }

        Map<Integer, List<Integer>> paths = new HashMap<>();
        Map<Integer, Map<Integer, Double>> winningBids = new HashMap<>();
        Map<Integer, Map<Integer, Integer>> winningUavs = new HashMap<>();

        for (FogDevice uav : uavs) {
            paths.put(uav.getId(), new ArrayList<>());
            Map<Integer, Double> bids = new HashMap<>();
            Map<Integer, Integer> winners = new HashMap<>();
            for (Tuple t : tasks) {
                bids.put(t.getCloudletId(), 0.0);
                winners.put(t.getCloudletId(), -1);
            }
            winningBids.put(uav.getId(), bids);
            winningUavs.put(uav.getId(), winners);
        }

        for (int iter = 0; iter < 15; iter++) {
            boolean changedBuild = false;

            // Phase 1: Bundle Construction
            for (FogDevice uav : uavs) {
                int uid = uav.getId();
                double baseU = getPathUtility(uid, paths.get(uid), taskMap, d_ut, d_tt);

                while (paths.get(uid).size() < uav.taskCapacity) {
                    Integer bestTid = null;
                    double bestBid = 0.0;

                    for (Tuple task : tasks) {
                        int tid = task.getCloudletId();
                        if (paths.get(uid).contains(tid)) continue;
                        
                        Double dist = d_ut.get(uid).get(tid);
                        if (dist == null || dist > uav.coverageRadius) continue;

                        List<Integer> newPath = new ArrayList<>(paths.get(uid));
                        newPath.add(tid);
                        double newU = getPathUtility(uid, newPath, taskMap, d_ut, d_tt);
                        double marginal = newU - baseU;

                        if (marginal > winningBids.get(uid).get(tid) + 1e-6 && marginal > bestBid) {
                            bestBid = marginal;
                            bestTid = tid;
                        }
                    }

                    if (bestTid != null) {
                        paths.get(uid).add(bestTid);
                        winningBids.get(uid).put(bestTid, bestBid);
                        winningUavs.get(uid).put(bestTid, uid);
                        baseU += bestBid;
                        changedBuild = true;
                    } else {
                        break;
                    }
                }
            }

            // Phase 2: Consensus
            boolean changedCons = false;
            for (FogDevice u1 : uavs) {
                for (FogDevice u2 : uavs) {
                    if (u1.getId() == u2.getId()) continue;
                    int uid1 = u1.getId(), uid2 = u2.getId();
                    
                    for (Tuple task : tasks) {
                        int tid = task.getCloudletId();
                        double b1 = winningBids.get(uid1).get(tid);
                        int w1 = winningUavs.get(uid1).get(tid);
                        double b2 = winningBids.get(uid2).get(tid);
                        int w2 = winningUavs.get(uid2).get(tid);

                        if (b2 > b1) {
                            winningBids.get(uid1).put(tid, b2);
                            winningUavs.get(uid1).put(tid, w2);
                            if (w1 == uid1) {
                                releaseTask(uid1, tid, paths, winningBids, winningUavs);
                                changedCons = true;
                            }
                        } else if (b2 == b1 && w2 > w1 && b2 > 0) {
                            // Tie-breaker
                            winningBids.get(uid1).put(tid, b2);
                            winningUavs.get(uid1).put(tid, w2);
                            if (w1 == uid1) {
                                releaseTask(uid1, tid, paths, winningBids, winningUavs);
                                changedCons = true;
                            }
                        }
                    }
                }
            }

            if (!changedBuild && !changedCons) break;
        }

        // Apply final assignments
        for (FogDevice uav : uavs) {
            int uid = uav.getId();
            for (int tid : paths.get(uid)) {
                Tuple task = taskMap.get(tid);
                if (task != null && task.assignedUavId == null && uav.acceptedTasks.size() < uav.taskCapacity) {
                    // Check latency constraint before absolute commit
                    double currentDataRate = 6250000.0;
                    double uavMips = uav.getHost().getTotalMips();
                    double totalMiInQueue = 0.0;
                    for (Tuple t : uav.acceptedTasks) totalMiInQueue += t.getCloudletLength();
                    
                    double transmissionDelayMs = (task.getCloudletFileSize() / currentDataRate) * 1000.0;
                    double predictedQueueDelayMs = ((totalMiInQueue + task.getCloudletLength()) / uavMips) * 1000.0;
                    
                    if (transmissionDelayMs + predictedQueueDelayMs <= task.tolerantLatency) {
                        uav.acceptedTasks.add(task);
                        task.assignedUavId = uid;
                        GlobalState.accumulatedThroughput += task.tupleDataSize;
                    }
                }
            }
        }
    }

    private void releaseTask(int uid, int tid, Map<Integer, List<Integer>> paths, 
                             Map<Integer, Map<Integer, Double>> winningBids, 
                             Map<Integer, Map<Integer, Integer>> winningUavs) {
        List<Integer> path = paths.get(uid);
        int idx = path.indexOf(tid);
        if (idx != -1) {
            for (int i = idx; i < path.size(); i++) {
                int r = path.get(i);
                winningBids.get(uid).put(r, 0.0);
                winningUavs.get(uid).put(r, -1);
            }
            paths.put(uid, new ArrayList<>(path.subList(0, idx)));
        }
    }

    private double getPathUtility(int uid, List<Integer> path, Map<Integer, Tuple> tasks, 
                                  Map<Integer, Map<Integer, Double>> d_ut, 
                                  Map<Integer, Map<Integer, Double>> d_tt) {
        if (path == null || path.isEmpty()) return 0.0;
        double currentTime = 0.0;
        double total = 0.0;
        Integer prev = null;

        for (int tid : path) {
            Tuple task = tasks.get(tid);
            Double dist = (prev == null) ? d_ut.get(uid).get(tid) : d_tt.get(prev).get(tid);
            if (dist == null) dist = 0.0;
            
            currentTime += dist / v;
            if (currentTime > task.tolerantLatency) return -1e9;
            total += Math.pow(phi, currentTime) * Math.exp(-beta * d_ut.get(uid).get(tid));
            currentTime += delta;
            prev = tid;
        }
        return total;
    }

    private FogDevice getMdForTask(Tuple task) {
        try {
            int srcId = task.getSourceDeviceId();
            return (FogDevice) org.cloudbus.cloudsim.core.CloudSim.getEntity(srcId);
        } catch (Exception e) {
            return null;
        }
    }
}
