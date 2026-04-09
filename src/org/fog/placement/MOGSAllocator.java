package org.fog.placement;

import java.util.*;

public class MOGSAllocator {

    /**
     * Executes the Many-to-One Gale-Shapley matching for UAV task offloading.
     * * @param tasks           List of all task IDs (Proposers)
     * @param uavs            List of all UAV IDs (Acceptors)
     * @param taskPrefs       Map of Task -> List of preferred UAVs (ranked highest to lowest)
     * @param uavPrefs        Map of UAV -> List of preferred Tasks (ranked highest to lowest)
     * @param uavCapacities   Map of UAV -> Maximum number of tasks it can handle
     * @return                A Map of Task -> Assigned UAV
     */
    public static Map<String, String> matchTasksToUAVs(
            List<String> tasks, 
            List<String> uavs,
            Map<String, List<String>> taskPrefs, 
            Map<String, List<String>> uavPrefs,
            Map<String, Integer> uavCapacities) {

        // Queue of tasks that are currently unassigned and still have UAVs to propose to
        Queue<String> freeTasks = new LinkedList<>(tasks);
        
        // Keep track of which UAV a task has proposed to next (index in preference list)
        Map<String, Integer> nextProposalIndex = new HashMap<>();
        for (String task : tasks) {
            nextProposalIndex.put(task, 0);
        }

        // The final matching: UAV -> List of assigned tasks
        Map<String, List<String>> uavAssignments = new HashMap<>();
        for (String uav : uavs) {
            uavAssignments.put(uav, new ArrayList<>());
        }

        // The reverse lookup matching: Task -> Assigned UAV
        Map<String, String> taskAssignments = new HashMap<>();

        while (!freeTasks.isEmpty()) {
            String task = freeTasks.poll();
            List<String> preferences = taskPrefs.get(task);
            
            // Safety check: if Python hasn't sent preferences yet, or task isn't in the list
            if (preferences == null || preferences.isEmpty()) {
                continue; 
            }
            
            int prefIndex = nextProposalIndex.getOrDefault(task, 0);

            // If the task has proposed to all UAVs on its list, it remains unassigned
            if (prefIndex >= preferences.size()) {
                continue; 
            }

            // Get the next most preferred UAV
            String targetUAV = preferences.get(prefIndex);
            nextProposalIndex.put(task, prefIndex + 1); // Increment for next time

            List<String> currentAssignments = uavAssignments.get(targetUAV);
            
            // Default capacity to 1 if not specified
            int capacity = uavCapacities.getOrDefault(targetUAV, 1);

            if (currentAssignments.size() < capacity) {
                // UAV has free space: Accept the task automatically
                currentAssignments.add(task);
                taskAssignments.put(task, targetUAV);
            } else {
                // UAV is full: It must compare the new task against its current worst task
                String worstAssignedTask = getWorstTask(targetUAV, currentAssignments, uavPrefs);
                
                if (uavPrefersNewTask(targetUAV, task, worstAssignedTask, uavPrefs)) {
                    // Reject the worst task
                    currentAssignments.remove(worstAssignedTask);
                    taskAssignments.remove(worstAssignedTask);
                    freeTasks.add(worstAssignedTask); // Send back to the free pool

                    // Accept the new task
                    currentAssignments.add(task);
                    taskAssignments.put(task, targetUAV);
                } else {
                    // UAV rejects the new task; it goes back into the free pool to propose to its next choice
                    freeTasks.add(task);
                }
            }
        }
        
        return taskAssignments;
    }

    // Helper: Find the least preferred task currently assigned to the UAV
    private static String getWorstTask(String uav, List<String> assignedTasks, Map<String, List<String>> uavPrefs) {
        List<String> preferences = uavPrefs.get(uav);
        if (preferences == null) return assignedTasks.get(0); // Fallback if no prefs exist
        
        String worstTask = null;
        int worstRank = -1;

        for (String task : assignedTasks) {
            int rank = preferences.indexOf(task);
            // Higher index means lower preference. 
            // If rank is -1 (not in list), we treat it as the worst possible.
            if (rank == -1) return task; 
            if (rank > worstRank) {
                worstRank = rank;
                worstTask = task;
            }
        }
        return worstTask;
    }

    // Helper: Check if the UAV prefers the new proposing task over its current worst task
    private static boolean uavPrefersNewTask(String uav, String newTask, String worstTask, Map<String, List<String>> uavPrefs) {
        List<String> preferences = uavPrefs.get(uav);
        if (preferences == null) return false; // Fallback
        
        int newRank = preferences.indexOf(newTask);
        int worstRank = preferences.indexOf(worstTask);

        // A lower index means it's higher up on the preference list
        if (newRank == -1) return false; 
        if (worstRank == -1) return true;
        
        return newRank < worstRank;
    }
}