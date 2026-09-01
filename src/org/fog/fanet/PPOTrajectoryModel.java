package org.fog.fanet;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.List;
import org.fog.entities.FogDevice;

public class PPOTrajectoryModel implements TrajectoryModel {

    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;

    @Override
    public void start() {
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

    @Override
    public void stop() {
        try {
            if (out != null) out.println("CLOSE\n");
            if (socket != null) socket.close();
            System.out.println("Disconnected from Python PPO Server.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void updateTrajectories(List<FogDevice> uavs, List<FogDevice> mds) {
        if (out == null || in == null) {
            System.out.println("No connection to Python. Skipping trajectory update...");
            return;
        }

        try {
            StringBuilder uavJsonBuilder = new StringBuilder();
            uavJsonBuilder.append("[");
            for (int i = 0; i < uavs.size(); i++) {
                FogDevice uav = uavs.get(i);
                
                // Calculate MDs in range using 8-sector mapping
                int[] sectors = new int[8];
                for (FogDevice md : mds) {
                    double dx = md.x_coord - uav.x_coord;
                    double dy = md.y_coord - uav.y_coord;
                    double dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist <= uav.coverageRadius) {
                        double angle = Math.atan2(dy, dx);
                        if (angle < 0) angle += 2 * Math.PI;
                        int sector = (int) (angle / (Math.PI / 4.0));
                        if (sector >= 8) sector = 7;
                        sectors[sector]++;
                    }
                }
                
                uavJsonBuilder.append(String.format(
                    "{\"id\": \"%s\", \"x\": %f, \"y\": %f, \"z\": %f, \"load\": %d, \"mds_n\": %d, \"mds_ne\": %d, \"mds_e\": %d, \"mds_se\": %d, \"mds_s\": %d, \"mds_sw\": %d, \"mds_w\": %d, \"mds_nw\": %d}",
                    uav.getName(), uav.x_coord, uav.y_coord, 50.0, uav.acceptedTasks.size(),
                    sectors[0], sectors[1], sectors[2], sectors[3], sectors[4], sectors[5], sectors[6], sectors[7]
                ));
                if (i < uavs.size() - 1) uavJsonBuilder.append(", ");
            }
            uavJsonBuilder.append("]");

            String jsonRequest = String.format(
                "{\"reward\": %f, \"uavs\": %s}",
                GlobalState.accumulatedThroughput, uavJsonBuilder.toString()
            );

            // Reset reward for next step
            GlobalState.accumulatedThroughput = 0.0;

            out.println(jsonRequest);
            String response = in.readLine();
            
            if (response != null && response.contains("uav_actions")) {
                // Parse rudimentary JSON since we don't have a JSON library easily accessible
                String[] parts = response.split("\\{");
                for (String part : parts) {
                    if (part.contains("\"id\"")) {
                        String id = extractJsonString(part, "id");
                        double angle = extractJsonDouble(part, "angle");
                        double distance = extractJsonDouble(part, "distance");
                        
                        for (FogDevice uav : uavs) {
                            if (uav.getName().equals(id)) {
                                uav.x_coord += distance * Math.cos(angle);
                                uav.y_coord += distance * Math.sin(angle);
                                // Boundary enforcement
                                uav.x_coord = Math.max(0.0, Math.min(2000.0, uav.x_coord));
                                uav.y_coord = Math.max(0.0, Math.min(2000.0, uav.y_coord));
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
    
    private String extractJsonString(String json, String key) {
        String search = "\"" + key + "\": \"";
        int start = json.indexOf(search);
        if (start == -1) return "";
        start += search.length();
        int end = json.indexOf("\"", start);
        return json.substring(start, end);
    }
    
    private double extractJsonDouble(String json, String key) {
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
