package org.fog.fanet;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.List;
import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;

public class MarlClient {
    private static Socket socket;
    private static PrintWriter out;
    private static BufferedReader in;

    public static void connect() {
        try {
            System.out.println("Attempting to connect to MARL Allocation Server on port 5501...");
            socket = new Socket("127.0.0.1", 5501);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            System.out.println("SUCCESS: Connected to MARL Allocation Server!");
        } catch (Exception e) {
            System.err.println("FAILED: Could not connect to MARL Allocation Server.");
        }
    }

    public static void disconnect() {
        try {
            if (out != null) out.println("CLOSE\n");
            if (socket != null) socket.close();
            System.out.println("Disconnected from MARL Allocation Server.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static String getAssignments(List<Tuple> tasks, List<FogDevice> uavs) {
        if (out == null || in == null) {
            return null;
        }

        try {
            StringBuilder json = new StringBuilder();
            json.append("{\"tasks\": [");
            for (int i = 0; i < tasks.size(); i++) {
                Tuple t = tasks.get(i);
                FogDevice md = getMdForTask(t);
                double x = md != null ? md.x_coord : 0.0;
                double y = md != null ? md.y_coord : 0.0;
                json.append(String.format("{\"id\": %d, \"comp\": %f, \"lat\": %f, \"x\": %f, \"y\": %f}",
                        t.getCloudletId(), (double) t.getCloudletLength(), t.tolerantLatency, x, y));
                if (i < tasks.size() - 1) json.append(", ");
            }
            json.append("], \"uavs\": [");
            for (int i = 0; i < uavs.size(); i++) {
                FogDevice u = uavs.get(i);
                double speed = u.getHost().getTotalMips(); // Approx processing speed
                json.append(String.format("{\"id\": %d, \"active\": true, \"batt\": %f, \"cap\": %d, \"rem_cap\": %d, \"speed\": %f, \"x\": %f, \"y\": %f, \"radius\": %f}",
                        u.getId(), u.getEnergyConsumption(), u.taskCapacity, u.taskCapacity - u.acceptedTasks.size(), speed, u.x_coord, u.y_coord, u.coverageRadius));
                if (i < uavs.size() - 1) json.append(", ");
            }
            json.append("]}");

            out.println(json.toString());
            return in.readLine();
        } catch (Exception e) {
            System.err.println("Error querying MARL server: " + e.getMessage());
            return null;
        }
    }

    private static FogDevice getMdForTask(Tuple task) {
        try {
            int srcId = task.getSourceDeviceId();
            return (FogDevice) org.cloudbus.cloudsim.core.CloudSim.getEntity(srcId);
        } catch (Exception e) {
            return null;
        }
    }
}
