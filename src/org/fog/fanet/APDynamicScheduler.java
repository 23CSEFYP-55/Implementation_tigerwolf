package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.ArrayList;
import java.util.List;

/**
 * APDynamicScheduler (Adaptive-Predictive Dynamic Repair Scheduler)
 *
 * Combines:
 * 1. State-cached persistent bindings in AssignmentState with selective invalidation
 * 2. Slack-Prioritized Min-Heap queue for servicing urgent tasks first
 * 3. Multi-objective utility incorporating processing speed, battery headroom, and load equity
 * 4. Pareto-optimal cooperative P2P offloading for saturated edge nodes
 */
public class APDynamicScheduler implements TaskScheduler {

    private final AssignmentState assignmentState;
    private final APRepairEngine repairEngine;

    // Metrics Tracking
    private long totalRepairQueueLength = 0;
    private long totalInvalidAssignments = 0;
    private long totalReassignedTasks = 0;
    private long schedulingCycles = 0;

    public APDynamicScheduler() {
        this.assignmentState = new AssignmentState();
        this.repairEngine = new APRepairEngine(this.assignmentState);
    }

    @Override
    public void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs) {
        this.schedulingCycles++;

        // STEP 1: Refresh UAV state and invalidate disconnected assignments
        List<Tuple> invalidTasks = repairEngine.refreshUAVState(uavs);
        this.totalInvalidAssignments += invalidTasks.size();

        // STEP 2: Construct unified repair candidate list (invalidated + newly arrived)
        List<Tuple> candidateList = new ArrayList<>(invalidTasks.size() + tasks.size());
        candidateList.addAll(invalidTasks);
        candidateList.addAll(tasks);

        this.totalRepairQueueLength += candidateList.size();

        if (candidateList.isEmpty()) {
            return;
        }

        // STEP 3: Assign tasks using Slack-Prioritized Min-Heap & Pareto P2P offloading
        long reassignedThisCycle = repairEngine.assignNewTasks(candidateList, uavs);
        this.totalReassignedTasks += reassignedThisCycle;
    }

    public long getTotalRepairQueueLength() { return totalRepairQueueLength; }
    public long getTotalInvalidAssignments() { return totalInvalidAssignments; }
    public long getTotalReassignedTasks() { return totalReassignedTasks; }
    public long getSchedulingCycles() { return schedulingCycles; }
}
