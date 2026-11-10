package org.fog.fanet;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import org.fog.entities.FogDevice;

/**
 * TACC-AAV (Task Allocation with Communication Coordination for AAV-Assisted Edge Computing)
 * Trajectory Model.
 * Communicates with the Python TACC-AAV socket server (port 5530).
 */
public class TACCAAVTrajectoryModel implements TrajectoryModel {

    private static final int PORT = 5530;
    private static final double BOUND = 2000.0;

    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;
    private boolean connected = false;

    private final Map<String, Double> uavHeadings = new HashMap<>();

    @Override
    public void start() {
        try {
            System.out.println("Attempting to connect to TACC-AAV Server on port " + PORT + "...");
            socket = new Socket("127.0.0.1", PORT);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            connected = true;
            System.out.println("SUCCESS: Connected to TACC-AAV Trajectory Server!");
        } catch (Exception e) {
            connected = false;
            System.err.println("FAILED: Could not connect to TACC-AAV Trajectory Server: " + e.getMessage());
        }
    }

    @Override
    public void stop() {
        try {
            if (out != null) {
                out.println("CLOSE");
                out.close();
            }
            if (in != null) in.close();
            if (socket != null) socket.close();
            System.out.println("Disconnected from TACC-AAV Server.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void updateTrajectories(List<FogDevice> uavs, List<FogDevice> mds) {
        if (!connected || out == null || in == null) {
            return;
        }

        try {
            double now = org.cloudbus.cloudsim.core.CloudSim.clock();

            // Build UAV JSON
            StringBuilder uavJson = new StringBuilder("[");
            for (int i = 0; i < uavs.size(); i++) {
                FogDevice u = uavs.get(i);
                double heading = uavHeadings.getOrDefault(u.getName(), 0.0);
                double speed = u.getHost().getTotalMips() / 200.0;
                double maxBatt = 100000.0;
                double energy = Math.max(0.0, maxBatt - u.getEnergyConsumption());

                uavJson.append(String.format(Locale.US,
                    "{\"id\": \"%s\", \"x\": %f, \"y\": %f, \"batt\": %f, \"speed\": %f, \"heading\": %f, \"load\": %d}",
                    u.getName(), u.x_coord, u.y_coord, energy, speed, heading, u.acceptedTasks.size()
                ));
                if (i < uavs.size() - 1) uavJson.append(", ");
            }
            uavJson.append("]");

            // Build MD / Task JSON
            StringBuilder mdJson = new StringBuilder("[");
            int mdSample = Math.min(mds.size(), 100);
            for (int i = 0; i < mdSample; i++) {
                FogDevice m = mds.get(i);
                mdJson.append(String.format(Locale.US,
                    "{\"id\": %d, \"x\": %f, \"y\": %f, \"comp\": 2000.0, \"size\": 10.0, \"deadline\": 500.0}",
                    m.getId(), m.x_coord, m.y_coord
                ));
                if (i < mdSample - 1) mdJson.append(", ");
            }
            mdJson.append("]");

            String req = String.format(Locale.US, "{\"time\": %f, \"uavs\": %s, \"mds\": %s}",
                now, uavJson.toString(), mdJson.toString());

            out.println(req);
            String response = in.readLine();

            if (response != null && response.contains("uav_actions")) {
                parseAndApplyActions(response, uavs);
            }
        } catch (Exception e) {
            connected = false;
        }
    }

    private void parseAndApplyActions(String response, List<FogDevice> uavs) {
        String[] parts = response.split("\\{");
        for (String part : parts) {
            if (!part.contains("\"id\"")) continue;

            String id = extractJsonString(part, "id");
            double angle = extractJsonDouble(part, "angle");
            double distance = extractJsonDouble(part, "distance");

            if (!id.isEmpty()) {
                uavHeadings.put(id, angle);
                for (FogDevice uav : uavs) {
                    if (uav.getName().equals(id)) {
                        uav.x_coord += distance * Math.cos(angle);
                        uav.y_coord += distance * Math.sin(angle);
                        // Boundary clamping
                        uav.x_coord = Math.max(0.0, Math.min(BOUND, uav.x_coord));
                        uav.y_coord = Math.max(0.0, Math.min(BOUND, uav.y_coord));
                        break;
                    }
                }
            }
        }
    }

    private String extractJsonString(String json, String key) {
        String search = "\"" + key + "\": \"";
        int start = json.indexOf(search);
        if (start == -1) return "";
        start += search.length();
        int end = json.indexOf("\"", start);
        return (end > start) ? json.substring(start, end) : "";
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
