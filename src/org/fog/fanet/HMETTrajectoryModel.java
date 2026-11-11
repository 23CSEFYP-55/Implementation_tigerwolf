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
 * H-MET (Hierarchical Macro-Micro Elastic Trajectory Optimization)
 *
 * Implements a two-tier hierarchical trajectory control framework:
 * 1. Tier-1 Macro Planner:
 *    - Uses CAR Otsu spatial partitioning and Pointer-Network TSP permutations
 *      to anchor flight corridors along urban hubs and transit arteries.
 *    - Continuous cyclic tour wrap-around so UAVs maintain perpetual patrol coverage.
 * 2. Tier-2 Micro Elastic Steering:
 *    - Closed-loop micro-drift toward local ground demand clusters (d <= 1.5 * Rcov).
 *    - Dwell speed modulation (0.75x) over active demand clusters to extend contact windows.
 *    - Transit acceleration (1.20x) across sparse zones.
 */
public class HMETTrajectoryModel implements TrajectoryModel {

    private static final int CAR_PORT = 5510;
    private static final double MAX_DIST = 50.0;          // max movement per sync step (m)
    private static final double PLAN_INTERVAL = 2000.0;   // replan cadence (sim ms)
    private static final double BOUND = 2000.0;           // 2x2 km boundary
    private static final double ELASTIC_WEIGHT = 0.20;    // 80% macro TSP momentum, 20% micro task pull

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
            System.out.println("Attempting to connect to CAR Pointer-Network Server on port " + CAR_PORT + " for H-MET...");
            socket = new Socket("127.0.0.1", CAR_PORT);
            out = new PrintWriter(socket.getOutputStream(), true);
            in = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            connected = true;
            System.out.println("SUCCESS: Connected to CAR Pointer-Network Server for H-MET!");
        } catch (Exception e) {
            connected = false;
            System.err.println("FAILED: Could not connect to CAR Pointer-Network Server on port " + CAR_PORT);
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

        // Recompute the macro CAR partition + pointer-network plan periodically.
        if (now - lastPlanTime >= PLAN_INTERVAL) {
            lastPlanTime = now;
            try {
                plan(uavs, mds);
            } catch (Exception e) {
                System.err.println("H-MET replan error: " + e.getMessage());
            }
        }

        // Execute Hierarchical Macro-Micro Steering for each UAV
        for (FogDevice uav : uavs) {
            String id = uav.getName();
            double[] xs = wayXs.get(id);
            int pos = wayPos.getOrDefault(id, -1);
            if (xs == null || xs.length == 0 || pos < 0) continue;

            // Cyclic wrap-around so drones loop continuously across waypoints
            if (pos >= xs.length) {
                pos = 0;
                wayPos.put(id, 0);
            }

            double tx = xs[pos];
            double ty = wayYs.get(id)[pos];
            double dx = tx - uav.x_coord;
            double dy = ty - uav.y_coord;
            double dist = Math.sqrt(dx * dx + dy * dy);

            double reach = Math.max(uav.coverageRadius, 15.0);
            if (dist <= reach) {
                // Arrived at current macro waypoint: advance cyclically to next waypoint
                int nextPos = (pos + 1) % xs.length;
                wayPos.put(id, nextPos);
                continue;
            }

            // 1. Nominal Macro Velocity Vector
            double macroVx = (dist > 1e-4) ? (dx / dist) : 0.0;
            double macroVy = (dist > 1e-4) ? (dy / dist) : 0.0;

            // 2. Micro Elastic Task Attraction Vector
            double microFx = 0.0;
            double microFy = 0.0;
            int localMdCount = 0;
            double searchRadius = uav.coverageRadius * 1.4; // 350m perception horizon

            for (FogDevice md : mds) {
                double dMdX = md.x_coord - uav.x_coord;
                double dMdY = md.y_coord - uav.y_coord;
                double dMd = Math.sqrt(dMdX * dMdX + dMdY * dMdY);

                if (dMd <= searchRadius) {
                    localMdCount++;
                    double weight = 1.0 / (1.0 + dMd * 0.01);
                    microFx += (dMdX / (dMd + 1.0)) * weight;
                    microFy += (dMdY / (dMd + 1.0)) * weight;
                }
            }

            double microMag = Math.sqrt(microFx * microFx + microFy * microFy);
            double microVx = (microMag > 1e-4) ? (microFx / microMag) : 0.0;
            double microVy = (microMag > 1e-4) ? (microFy / microMag) : 0.0;

            // 3. Composite Steering Angle
            double combinedVx, combinedVy;
            if (localMdCount > 0 && microMag > 1e-4) {
                combinedVx = (1.0 - ELASTIC_WEIGHT) * macroVx + ELASTIC_WEIGHT * microVx;
                combinedVy = (1.0 - ELASTIC_WEIGHT) * macroVy + ELASTIC_WEIGHT * microVy;
            } else {
                combinedVx = macroVx;
                combinedVy = macroVy;
            }

            double steerAngle = Math.atan2(combinedVy, combinedVx);

            // 4. Contact-Window Speed Modulation
            double baseStep = stepDistance(id);
            double speedMultiplier = 1.0;
            if (localMdCount >= 2) {
                // Dwell mode: slightly slower speed (0.78x) to extend wireless contact duration
                speedMultiplier = 0.78;
            } else if (localMdCount == 0) {
                // Sprint mode: accelerate across empty transit space
                speedMultiplier = 1.15;
            }

            double move = Math.min(baseStep * speedMultiplier, dist);
            uav.x_coord = clamp(uav.x_coord + move * Math.cos(steerAngle));
            uav.y_coord = clamp(uav.y_coord + move * Math.sin(steerAngle));
        }
    }

    /**
     * H-MET planning stage: capability assessment -> adaptive clustering -> sequential
     * matching -> pointer-network tour ordering (request to Python server).
     */
    private void plan(List<FogDevice> uavs, List<FogDevice> mds) {
        if (uavs.isEmpty() || mds.isEmpty()) return;

        // Stage 1: UAV capability assessment
        List<CARRegionClustering.UavProfile> profiles = CARRegionClustering.assessUavCapabilities(uavs);

        // Stage 2: adaptive density/distance region partitioning with Otsu center selection
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
            System.err.println("H-MET matching error: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        // Ask the pointer network to order each UAV's cluster waypoints
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
        System.out.println("[H-MET] Placed hierarchical tours for " + rawPlans.size() + " UAVs (" + orders.size()
                + " server tours, " + clusters.size() + " clusters).");
    }

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
            System.err.println("H-MET pointer-network request error: " + e.getMessage());
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
