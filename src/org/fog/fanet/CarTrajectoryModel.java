package org.fog.fanet;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.fog.entities.FogDevice;

/**
 * CAR (Capability-aware Adaptive Route optimization) trajectory model — implements
 * the trajectory planning framework of:
 *
 *   "UAV Trajectory Optimization Based on Pointer Networks and Adaptive Region
 *    Partitioning" (Guo, Tang, Tan, Luo, Zhao — IEEE IoT Journal).
 *
 * The CAR pipeline (paper Algorithm 1) is executed as follows:
 *
 *   1. UAV capability assessment (Eq. 15) — CARRegionClustering.assessUavCapabilities().
 *   2. Adaptive density/distance region partitioning (Eqs. 16-17) with Otsu center
 *      selection — CARRegionClustering.partitionRegions().
 *   3. Sequential capability-cluster matching — each UAV owns one cluster.
 *   4. Pointer-network trajectory planning — each UAV's cluster waypoints are sent to
 *      the CAR pointer-network Python server (port 5510), which returns a visiting
 *      permutation Y (Eq. 18). The UAV then follows that order, moving <= MAX_DIST
 *      per sync step (like the PPO model's action space).
 *
 * The plan is (re-)computed every PLAN_INTERVAL ms so the trajectory adapts to the
 * current region/MD distribution.
 */
public class CarTrajectoryModel implements TrajectoryModel {

    private static final int CAR_PORT = 5510;
    private static final double MAX_DIST = 50.0;          // max movement per sync step (m)
    private static final double PLAN_INTERVAL = 2000.0;   // replan cadence (sim ms)
    private static final double BOUND = 2000.0;           // 2x2 km boundary

    private Socket socket;
    private PrintWriter out;
    private BufferedReader in;

    private final Map<String, double[]> wayXs = new HashMap<>();
    private final Map<String, double[]> wayYs = new HashMap<>();
    private final Map<String, Integer> wayPos = new HashMap<>();
    private final Map<String, Double> uavSpeed = new HashMap<>();

    private boolean connected = false;
    private double lastPlanTime = -PLAN_INTERVAL - 1.0;

    @Override
    public void start() {
        try {
            System.out.println("Attempting to connect to CAR Pointer-Network Server on port " + CAR_PORT + "...");
            socket = new Socket("127.0.0.1", CAR_PORT);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            connected = true;
            System.out.println("SUCCESS: Connected to CAR Pointer-Network Server!");
        } catch (Exception e) {
            connected = false;
            System.err.println("FAILED: Could not connect to CAR Pointer-Network Server.");
        }
    }

    @Override
    public void stop() {
        try {
            if (out != null) out.println("CLOSE\n");
            if (socket != null) socket.close();
            System.out.println("Disconnected from CAR Pointer-Network Server.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void updateTrajectories(List<FogDevice> uavs, List<FogDevice> mds) {
        if (!connected || out == null || in == null) {
            return;
        }

        double now = org.cloudbus.cloudsim.core.CloudSim.clock();

        // Recompute the CAR partition + pointer-network plan periodically.
        if (now - lastPlanTime >= PLAN_INTERVAL) {
            lastPlanTime = now;
            try {
                plan(uavs, mds);
            } catch (Exception e) {
                System.err.println("CAR replan error: " + e.getMessage());
            }
        }

        // Follow the planned waypoint order: move each UAV toward its current target.
        for (FogDevice uav : uavs) {
            String id = uav.getName();
            double[] xs = wayXs.get(id);
            int pos = wayPos.getOrDefault(id, -1);
            if (xs == null || pos < 0 || pos >= xs.length) continue;

            double tx = xs[pos];
            double ty = wayYs.get(id)[pos];
            double dx = tx - uav.x_coord;
            double dy = ty - uav.y_coord;
            double dist = Math.sqrt(dx * dx + dy * dy);

            double reach = Math.max(uav.coverageRadius, 10.0);
            if (dist <= reach) {
                // Arrived: advance to the next region in the visitation order.
                uav.x_coord = clamp(tx);
                uav.y_coord = clamp(ty);
                wayPos.put(id, pos + 1);
                continue;
            }

            double step = stepDistance(id);
            double move = Math.min(step, dist);
            double angle = Math.atan2(dy, dx);
            uav.x_coord = clamp(uav.x_coord + move * Math.cos(angle));
            uav.y_coord = clamp(uav.y_coord + move * Math.sin(angle));
        }
    }

    /**
     * CAR planning stage: capability assessment -> adaptive clustering -> sequential
     * matching -> pointer-network tour ordering (request to Python server).
     */
    private void plan(List<FogDevice> uavs, List<FogDevice> mds) {
        if (uavs.isEmpty() || mds.isEmpty()) return;

        // Stage 1: UAV capability assessment (Eq. 15).
        List<CARRegionClustering.UavProfile> profiles = CARRegionClustering.assessUavCapabilities(uavs);

        // Stage 2: adaptive density/distance region partitioning (Eqs. 16-17).
        List<CARRegionClustering.Cluster> clusters =
                CARRegionClustering.partitionRegions(mds, uavs, profiles);

        // Stage 3 + 4: sequential matching, then pointer-network tour ordering.
        for (FogDevice uav : uavs) {
            uavSpeed.put(uav.getName(), profiles.get(uavs.indexOf(uav)).speedVm);
        }

        List<CARRegionClustering.Plan> rawPlans;
        try {
            rawPlans = CARRegionClustering.matchAndPlan(uavs, mds, profiles, clusters);
        } catch (Exception e) {
            System.err.println("CAR matching error: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        // Ask the pointer network to order each UAV's cluster waypoints (Eq. 18-21).
        Map<String, int[]> orders = requestTours(rawPlans);

        for (CARRegionClustering.Plan raw : rawPlans) {
            String id = raw.uav.getName();
            int[] order = orders.get(id);
            if (order == null) {
                order = new int[raw.wayXs.length];
                for (int i = 0; i < order.length; i++) order[i] = i;
            }
            double[] xs = new double[order.length];
            double[] ys = new double[order.length];
            for (int i = 0; i < order.length; i++) {
                int r = order[i];
                if (r < 0 || r >= raw.wayXs.length) continue;
                xs[i] = raw.wayXs[r];
                ys[i] = raw.wayYs[r];
            }
            wayXs.put(id, xs);
            wayYs.put(id, ys);
            wayPos.put(id, 0);
        }
        System.out.println("[CAR] Placed tours for " + rawPlans.size() + " UAVs (" + orders.size()
                + " server tours, " + clusters.size() + " clusters).");
    }

    /** Sends all UAV clusters to the CAR server and parses the returned tours. */
    private Map<String, int[]> requestTours(List<CARRegionClustering.Plan> plans) {
        Map<String, int[]> tours = new HashMap<>();
        if (plans.isEmpty()) return tours;
        try {
            StringBuilder req = new StringBuilder();
            req.append("{\"plan\": [");
            for (int i = 0; i < plans.size(); i++) {
                CARRegionClustering.Plan p = plans.get(i);
                req.append(String.format("{\"id\": \"%s\", \"origin\": [%f, %f], \"regions\": [",
                        p.uav.getName(), p.uav.x_coord, p.uav.y_coord));
                for (int r = 0; r < p.wayXs.length; r++) {
                    req.append(String.format("[%f, %f]", p.wayXs[r], p.wayYs[r]));
                    if (r < p.wayXs.length - 1) req.append(", ");
                }
                req.append("]}");
                if (i < plans.size() - 1) req.append(", ");
            }
            req.append("]}");

            out.println(req.toString());
            String response = in.readLine();
            if (response != null && response.contains("tours")) {
                String[] parts = response.split("\\{");
                for (String part : parts) {
                    if (!part.contains("\"id\"")) continue;
                    String id = extractJsonString(part, "id").trim();
                    int[] order = extractJsonIntArray(part, "order");
                    if (!id.isEmpty() && order.length > 0) {
                        tours.put(id, order);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("CAR pointer-network request error: " + e.getMessage());
        }
        return tours;
    }

    private double stepDistance(String id) {
        double v = uavSpeed.getOrDefault(id, 12.0);
        return MAX_DIST * (v / 22.0);
    }

    private static double clamp(double v) {
        return Math.max(0.0, Math.min(BOUND, v));
    }

    private String extractJsonString(String json, String key) {
        String search = "\"" + key + "\": \"";
        int start = json.indexOf(search);
        if (start == -1) return "";
        start += search.length();
        int end = json.indexOf("\"", start);
        return json.substring(start, end);
    }

    /** Extracts an integer array value, e.g. "order": [1, 2, 3]. */
    private int[] extractJsonIntArray(String json, String key) {
        String search = "\"" + key + "\": [";
        int start = json.indexOf(search);
        if (start == -1) return new int[0];
        start += search.length();
        int end = json.indexOf("]", start);
        if (end == -1) return new int[0];
        String[] bits = json.substring(start, end).split(",");
        List<Integer> order = new ArrayList<>();
        for (String bit : bits) {
            String trimmed = bit.trim();
            if (trimmed.matches("\\d+")) order.add(Integer.parseInt(trimmed));
        }
        int[] arr = new int[order.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = order.get(i);
        return arr;
    }
}