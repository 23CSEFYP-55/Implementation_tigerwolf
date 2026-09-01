package org.fog.fanet;

import org.fog.entities.Tuple;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Stores and manages the persistent assignment state for the DynamicRepairScheduler.
 */
public class AssignmentState {
    // Maps UAV ID to the list of tasks currently assigned to it
    private Map<Integer, List<Tuple>> currentAssignments;

    public AssignmentState() {
        this.currentAssignments = new HashMap<>();
    }

    /**
     * Adds a task to a specific UAV's assignment list.
     */
    public void addTask(Integer uavId, Tuple task) {
        currentAssignments.computeIfAbsent(uavId, k -> new ArrayList<>()).add(task);
    }

    /**
     * Removes a task from a specific UAV's assignment list.
     */
    public void removeTask(Integer uavId, Tuple task) {
        if (currentAssignments.containsKey(uavId)) {
            currentAssignments.get(uavId).remove(task);
        }
    }

    /**
     * Gets the list of tasks currently assigned to a given UAV.
     */
    public List<Tuple> getAssignments(Integer uavId) {
        return currentAssignments.getOrDefault(uavId, new ArrayList<>());
    }

    /**
     * Invalidates (removes) all assignments for a given UAV and returns them for repair.
     */
    public List<Tuple> invalidateAssignments(Integer uavId) {
        List<Tuple> invalidatedTasks = currentAssignments.remove(uavId);
        return invalidatedTasks != null ? invalidatedTasks : new ArrayList<>();
    }

    /**
     * Helper to get the underlying map for refreshing state.
     */
    public Map<Integer, List<Tuple>> getAllAssignments() {
        return currentAssignments;
    }
}
