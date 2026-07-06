package org.fog.fanet;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.List;
import org.fog.entities.FogDevice;
import org.cloudbus.cloudsim.core.CloudSim;

public class PreferenceBuilder {

    private static Socket socket;
    private static PrintWriter out;
    private static BufferedReader in;
    
    // Reward accumulated since last step
    public static double accumulatedThroughput = 0.0;

    public static void connectToPPO() {
        try {
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
            if (out != null) out.println("CLOSE\n"); 
            if (socket != null) socket.close();
            System.out.println("Disconnected from Python PPO Server.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void updateTrajectories(List<FogDevice> uavs, List<FogDevice> mds) {
        if (out == null || in == null) {
            System.out.println("No connection to Python. Skipping trajectory update...");
            return;
        }

        try {
            StringBuilder uavJsonBuilder = new StringBuilder();
            uavJsonBuilder.append("[");
            for (int i = 0; i < uavs.size(); i++) {
                FogDevice uav = uavs.get(i);
                
                // Calculate MDs in range
                int mdsInRange = 0;
                for (FogDevice md : mds) {
                    double dist = Math.sqrt(Math.pow(uav.x_coord - md.x_coord, 2) + Math.pow(uav.y_coord - md.y_coord, 2));
                    if (dist <= uav.coverageRadius) {
                        mdsInRange++;
                    }
                }
                
                uavJsonBuilder.append(String.format(
                    "{\"id\": \"%s\", \"x\": %f, \"y\": %f, \"z\": %f, \"load\": %d, \"mds_in_range\": %d}",
                    uav.getName(), uav.x_coord, uav.y_coord, 50.0, uav.acceptedTasks.size(), mdsInRange
                ));
                if (i < uavs.size() - 1) uavJsonBuilder.append(", ");
            }
            uavJsonBuilder.append("]");

            String jsonRequest = String.format(
                "{\"reward\": %f, \"uavs\": %s}",
                accumulatedThroughput, uavJsonBuilder.toString()
            );

            // Reset reward for next step
            accumulatedThroughput = 0.0;

            out.println(jsonRequest);
            String response = in.readLine();
            
            if (response != null && response.contains("uav_actions")) {
                // Parse rudimentary JSON since we don't have a JSON library easily accessible
                String[] parts = response.split("\\{");
                for (String part : parts) {
                    if (part.contains("\"id\"")) {
                        String id = extractJsonString(part, "id");
                        double dx = extractJsonDouble(part, "dx");
                        double dy = extractJsonDouble(part, "dy");
                        double dz = extractJsonDouble(part, "dz");
                        
                        for (FogDevice uav : uavs) {
                            if (uav.getName().equals(id)) {
                                uav.x_coord += dx;
                                uav.y_coord += dy;
                                // Boundary enforcement
                                uav.x_coord = Math.max(0, Math.min(2000, uav.x_coord));
                                uav.y_coord = Math.max(0, Math.min(2000, uav.y_coord));
                                break;
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error communicating with Python server: " + e.getMessage());
        }
    }
    
    private static String extractJsonString(String json, String key) {
        String search = "\"" + key + "\": \"";
        int start = json.indexOf(search);
        if (start == -1) return "";
        start += search.length();
        int end = json.indexOf("\"", start);
        return json.substring(start, end);
    }
    
    private static double extractJsonDouble(String json, String key) {
        String search = "\"" + key + "\": ";
        int start = json.indexOf(search);
        if (start == -1) return 0.0;
        start += search.length();
        int end = json.indexOf(",", start);
        if (end == -1) end = json.indexOf("}", start);
        if (end == -1) return 0.0;
        try {
            return Double.parseDouble(json.substring(start, end).trim());
        } catch (Exception e) {
            return 0.0;
        }
    }
}