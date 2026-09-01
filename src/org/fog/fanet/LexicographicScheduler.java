package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.*;

public class LexicographicScheduler implements TaskScheduler {

    @Override
    public void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs) {
        if (tasks.isEmpty() || uavs.isEmpty()) return;

        Map<Integer, FogDevice> uavMap = new HashMap<>();
        for (FogDevice uav : uavs) {
            uavMap.put(uav.getId(), uav);
        }

        // 1. Tasks (Ground Devices) rank UAVs
        Map<Integer, List<Integer>> taskPreferences = new HashMap<>();
        for (Tuple task : tasks) {
            Map<Integer, Double> uavScores = new HashMap<>();
            FogDevice md = getMdForTask(task);
            if (md != null) {
                for (FogDevice uav : uavs) {
                    double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));
                    if (dist <= uav.coverageRadius) {
                        double r_up = 1.0 / (1.0 + dist);
                        double r_down = 1.0 / (1.0 + dist);
                        double load_penalty = 1.0 / (1.0 + uav.acceptedTasks.size());
                        double score = 0.4 * r_up + 0.4 * r_down + 0.2 * load_penalty;
                        uavScores.put(uav.getId(), score);
                    }
                }
            }
            List<Integer> sortedUavs = new ArrayList<>(uavScores.keySet());
            sortedUavs.sort((id1, id2) -> Double.compare(uavScores.get(id2), uavScores.get(id1)));
            taskPreferences.put(task.getCloudletId(), sortedUavs);
            task.currentProposalIndex = 0;
            task.assignedUavId = null;
        }

        // 2. UAVs rank Tasks
        Map<Integer, Map<Integer, Integer>> uavTaskRanks = new HashMap<>(); // uavId -> {taskId -> rank}
        for (FogDevice uav : uavs) {
            List<Tuple> sortedTasks = new ArrayList<>(tasks);
            sortedTasks.sort((t1, t2) -> {
                FogDevice md1 = getMdForTask(t1);
                FogDevice md2 = getMdForTask(t2);
                double d1 = md1 != null ? Math.sqrt(Math.pow(uav.x_coord - md1.x_coord, 2) + Math.pow(uav.y_coord - md1.y_coord, 2)) : Double.MAX_VALUE;
                double d2 = md2 != null ? Math.sqrt(Math.pow(uav.x_coord - md2.x_coord, 2) + Math.pow(uav.y_coord - md2.y_coord, 2)) : Double.MAX_VALUE;
                return Double.compare(d1, d2); // Ascending distance
            });
            Map<Integer, Integer> ranks = new HashMap<>();
            for (int i = 0; i < sortedTasks.size(); i++) {
                ranks.put(sortedTasks.get(i).getCloudletId(), i);
            }
            uavTaskRanks.put(uav.getId(), ranks);
        }

        // 3. Deferred Acceptance
        Queue<Tuple> unassignedTasks = new LinkedList<>(tasks);
        Map<Integer, Set<Tuple>> uavProposals = new HashMap<>();
        for (FogDevice uav : uavs) {
            uavProposals.put(uav.getId(), new HashSet<>(uav.acceptedTasks));
        }

        while (!unassignedTasks.isEmpty()) {
            Tuple task = unassignedTasks.poll();
            List<Integer> prefs = taskPreferences.get(task.getCloudletId());
            
            if (prefs == null || task.currentProposalIndex >= prefs.size()) {
                // Rejected by everyone
                continue;
            }

            int targetUavId = prefs.get(task.currentProposalIndex);
            task.currentProposalIndex++;
            FogDevice targetUav = uavMap.get(targetUavId);

            if (targetUav != null) {
                Set<Tuple> currentProps = uavProposals.get(targetUavId);
                currentProps.add(task);

                // Lexicographic choice: Sort all current proposals by UAV's preference
                List<Tuple> propList = new ArrayList<>(currentProps);
                Map<Integer, Integer> prefOrder = uavTaskRanks.get(targetUavId);
                propList.sort(Comparator.comparingInt(t -> prefOrder.getOrDefault(t.getCloudletId(), Integer.MAX_VALUE)));

                // Greedily accept top tasks that meet capacity and latency constraints
                Set<Tuple> chosen = new HashSet<>();
                double totalMiInQueue = 0.0;
                double uavMips = targetUav.getHost().getTotalMips();
                double currentDataRate = 6250000.0; // 50 Mbps

                for (Tuple candidate : propList) {
                    if (chosen.size() < targetUav.taskCapacity) {
                        double transmissionDelayMs = (candidate.getCloudletFileSize() / currentDataRate) * 1000.0;
                        double predictedQueueDelayMs = ((totalMiInQueue + candidate.getCloudletLength()) / uavMips) * 1000.0;
                        if (transmissionDelayMs + predictedQueueDelayMs <= candidate.tolerantLatency) {
                            chosen.add(candidate);
                            totalMiInQueue += candidate.getCloudletLength();
                        }
                    }
                }

                uavProposals.put(targetUavId, chosen);

                // Any task in currentProps but NOT in chosen is rejected, put back in queue
                for (Tuple t : currentProps) {
                    if (!chosen.contains(t) && !targetUav.acceptedTasks.contains(t)) {
                        unassignedTasks.add(t);
                    }
                }
            } else {
                unassignedTasks.add(task);
            }
        }

        // 4. Finalize Assignments
        for (FogDevice uav : uavs) {
            Set<Tuple> finalTasks = uavProposals.get(uav.getId());
            for (Tuple t : finalTasks) {
                if (!uav.acceptedTasks.contains(t)) {
                    uav.acceptedTasks.add(t);
                    t.assignedUavId = uav.getId();
                    GlobalState.accumulatedThroughput += t.tupleDataSize;
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
}
