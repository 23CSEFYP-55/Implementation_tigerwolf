package org.fog.fanet;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;

public class PreferenceBuilder {

    private static Socket socket;
    private static PrintWriter out;
    private static BufferedReader in;
    
    // The tracker the Controller uses
    public static Map<String, Integer> uavQueueTracker = new HashMap<>();

    public static void connectToPPO() {
        try {
            // MATCHED TO PYTHON PORT 5500
            System.out.println("Attempting to connect to Python TF-PPO Server on port 5500...");
            socket = new Socket("127.0.0.1", 5500);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            System.out.println("SUCCESS: Connected to Python Server!");
        } catch (Exception e) {
            System.err.println("FAILED: Could not connect to Python Server.");
        }
    }

    public static void disconnectFromPPO() {
        try {
            if (out != null) {
                out.println("CLOSE\n"); 
            }
            if (socket != null) {
                socket.close();
            }
            System.out.println("Disconnected from Python PPO Server.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void buildPreferences(List<Tuple> tasks, List<FogDevice> uavs) {
        if (out == null || in == null) {
            System.out.println("No connection to Python. Skipping RL matching...");
            return;
        }

        try {
            // First, sync our tracker with the TRUE simulator state before the batch starts
            for (FogDevice uav : uavs) {
                uavQueueTracker.put(uav.getName(), uav.acceptedTasks.size());
            }

            // Process each task in the batch
            for (Tuple task : tasks) {
                
                // 1. Build UAV JSON dynamically INSIDE the loop so it has fresh queue numbers
                StringBuilder uavJsonBuilder = new StringBuilder();
                uavJsonBuilder.append("[");
                for (int i = 0; i < uavs.size(); i++) {
                    FogDevice uav = uavs.get(i);
                    int currentQueue = uavQueueTracker.getOrDefault(uav.getName(), 0);
                    
                    uavJsonBuilder.append(String.format(
                        "{\"id\": \"%s\", \"mips\": %f, \"queue_size\": %d, \"x\": %f, \"y\": %f}",
                        uav.getName(), uav.getHost().getTotalMips(), currentQueue, uav.x_coord, uav.y_coord
                    ));
                    if (i < uavs.size() - 1) uavJsonBuilder.append(", ");
                }
                uavJsonBuilder.append("]");

                // 2. Construct the JSON object
                String jsonRequest = String.format(
                    "{\"task_id\": %d, \"task_data_size\": %f, \"task_cpu_cycles\": %f, \"uavs\": %s}",
                    task.getCloudletId(), (double) task.getCloudletFileSize(), (double) task.getCloudletLength(), uavJsonBuilder.toString()
                );

                // 3. Send to Python
                out.println(jsonRequest);

                // 4. Receive the ranked response from Python
                String response = in.readLine();
                
                if (response != null && response.contains("ranked_uavs")) {
                    String arrayContent = response.substring(response.indexOf("[") + 1, response.indexOf("]"));
                    String[] rankedIds = arrayContent.replace("\"", "").replace(" ", "").split(",");
                    
                    if (rankedIds.length > 0) {
                        String winningUav = rankedIds[0];
                        task.assignedUavId = getDeviceIdByName(uavs, winningUav);
                        
                        // 5. INSTANT TRACKER UPDATE: Add +1 so the next task in this loop sees it!
                        int newQueueCount = uavQueueTracker.getOrDefault(winningUav, 0) + 1;
                        uavQueueTracker.put(winningUav, newQueueCount);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error communicating with Python server: " + e.getMessage());
        }
    }

    private static Integer getDeviceIdByName(List<FogDevice> uavs, String name) {
        for (FogDevice uav : uavs) {
            if (uav.getName().equals(name)) {
                return uav.getId();
            }
        }
        return null;
    }
}