package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.*;

public class MOGSMatching {

    public static void runMatching(List<Tuple> tasks, List<FogDevice> uavs) {
        Queue<Tuple> unassignedTasks = new LinkedList<>(tasks);

        // Map for quick UAV lookup by ID
        Map<Integer, FogDevice> uavMap = new HashMap<>();
        for (FogDevice uav : uavs) {
            uavMap.put(uav.getId(), uav);
        }

        while (!unassignedTasks.isEmpty()) {
            Tuple task = unassignedTasks.poll();

            // If task proposed to all UAVs and was rejected by all, mark as null (send to cloud)
            if (task.currentProposalIndex >= task.preferredUAVs.size()) {
                task.assignedUavId = null;
                continue;
            }

            // Task proposes to its current top choice
            int targetUavId = task.preferredUAVs.get(task.currentProposalIndex);
            FogDevice targetUav = uavMap.get(targetUavId);
            task.currentProposalIndex++;

            // UAV tentatively accepts the proposal
            targetUav.acceptedTasks.add(task);

            // UAV sorts its accepted tasks based on its own preferences (descending score)
            targetUav.acceptedTasks.sort((t1, t2) -> {
                int score1 = targetUav.taskScores.getOrDefault(t1.getCloudletId(), 0);
                int score2 = targetUav.taskScores.getOrDefault(t2.getCloudletId(), 0);
                return Integer.compare(score2, score1);
            });

            // Enforce UAV capacity limit
            while (targetUav.acceptedTasks.size() > targetUav.taskCapacity) {
                // Kick out the lowest ranked task (the last one in the sorted list)
                Tuple rejectedTask = targetUav.acceptedTasks.remove(targetUav.acceptedTasks.size() - 1);
                rejectedTask.assignedUavId = null;
                unassignedTasks.add(rejectedTask); // Goes back into queue to try its next choice
            }

            // Update assignment status for tasks that survived the cut
            for (Tuple accepted : targetUav.acceptedTasks) {
                accepted.assignedUavId = targetUav.getId();
            }
        }
    }
}