package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * A dynamic incremental assignment scheduler designed as an alternative to MOGSScheduler.
 * It uses dynamic matching maintenance to repair affected assignments and schedule new tasks,
 * rather than rebuilding the entire global assignment every cycle.
 */
public class DynamicRepairScheduler implements TaskScheduler {

    private AssignmentState assignmentState;
    private RepairEngine repairEngine;

    // Metrics Tracking
    private long totalRepairQueueLength = 0;
    private long totalInvalidAssignments = 0;
    private long totalReassignedTasks = 0;
    private long schedulingCycles = 0;

    public DynamicRepairScheduler() {
        this.assignmentState = new AssignmentState();
        this.repairEngine = new RepairEngine(this.assignmentState);
    }

    @Override
    public void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs) {
        this.schedulingCycles++;
        
        // STEP 1: Refresh UAV state and invalidate affected assignments
        List<Tuple> invalidTasks = repairEngine.refreshUAVState(uavs);
        this.totalInvalidAssignments += invalidTasks.size();

        // STEP 2: Construct the unified repair queue
        Queue<Tuple> repairQueue = new LinkedList<>(invalidTasks);
        
        // Add only newly arrived tasks to the queue (those not already tracked in assignment state)
        for (Tuple task : tasks) {
            // MOGSScheduler re-attempts all tasks in the list.
            // Since we persist state, we assume 'tasks' represents newly arrived or unassigned tasks in the waiting pool.
            repairQueue.add(task);
        }

        this.totalRepairQueueLength += repairQueue.size();

        if (repairQueue.isEmpty()) {
            return; // Nothing to schedule or repair
        }

        // STEP 3: Assign tasks incrementally
        long reassignedThisCycle = repairEngine.assignNewTasks(repairQueue, uavs);
        this.totalReassignedTasks += reassignedThisCycle;
        
        // Unassigned tasks (those left with task.assignedUavId == null) 
        // will simply be dropped by the Controller or retained if the event loop supports it, 
        // mirroring the previous MOGS behavior without requiring Gale-Shapley retry loops.
    }

    public long getTotalRepairQueueLength() { return totalRepairQueueLength; }
    public long getTotalInvalidAssignments() { return totalInvalidAssignments; }
    public long getTotalReassignedTasks() { return totalReassignedTasks; }
    public long getSchedulingCycles() { return schedulingCycles; }
}
