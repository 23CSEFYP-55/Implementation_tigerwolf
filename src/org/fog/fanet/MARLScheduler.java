package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;
import java.util.*;

public class MARLScheduler implements TaskScheduler {

    @Override
    public void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs) {
        if (tasks.isEmpty() || uavs.isEmpty()) return;

        // Call Python MARL Allocation Server
        String jsonResponse = MarlClient.getAssignments(tasks, uavs);
        if (jsonResponse == null || jsonResponse.trim().isEmpty() || jsonResponse.contains("error")) {
            System.err.println("MARL Allocation Server returned null/error. Skipping epoch.");
            return;
        }

        Map<Integer, FogDevice> uavMap = new HashMap<>();
        for (FogDevice uav : uavs) uavMap.put(uav.getId(), uav);

        Map<Integer, Tuple> taskMap = new HashMap<>();
        for (Tuple t : tasks) taskMap.put(t.getCloudletId(), t);

        // Parse JSON assignments manually: {"assignments": [{"task_id": 1, "uav_id": 3}, ...]}
        try {
            int startIdx = jsonResponse.indexOf("[");
            int endIdx = jsonResponse.lastIndexOf("]");
            if (startIdx != -1 && endIdx != -1 && endIdx > startIdx) {
                String arrayContent = jsonResponse.substring(startIdx + 1, endIdx);
                String[] objects = arrayContent.split("},");
                for (String obj : objects) {
                    if (obj.trim().isEmpty()) continue;
                    
                    // extract task_id
                    int tIdx = obj.indexOf("\"task_id\":");
                    if (tIdx == -1) continue;
                    int tComma = obj.indexOf(",", tIdx);
                    if (tComma == -1) tComma = obj.indexOf("}", tIdx);
                    if (tComma == -1) tComma = obj.length();
                    String tStr = obj.substring(tIdx + 10, tComma).trim().replaceAll("[^0-9]", "");
                    
                    // extract uav_id
                    int uIdx = obj.indexOf("\"uav_id\":");
                    if (uIdx == -1) continue;
                    int uComma = obj.indexOf(",", uIdx);
                    if (uComma == -1) uComma = obj.indexOf("}", uIdx);
                    if (uComma == -1) uComma = obj.length();
                    String uStr = obj.substring(uIdx + 9, uComma).trim().replaceAll("[^0-9]", "");
                    
                    if (!tStr.isEmpty() && !uStr.isEmpty()) {
                        int taskId = Integer.parseInt(tStr);
                        int uavId = Integer.parseInt(uStr);
                        
                        Tuple t = taskMap.get(taskId);
                        FogDevice u = uavMap.get(uavId);
                        
                        if (t != null && u != null) {
                            u.acceptedTasks.add(t);
                            t.assignedUavId = u.getId();
                            GlobalState.accumulatedThroughput += t.tupleDataSize;
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to parse MARL JSON response: " + e.getMessage());
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
