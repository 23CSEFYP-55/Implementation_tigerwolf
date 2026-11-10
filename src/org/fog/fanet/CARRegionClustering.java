package org.fog.fanet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.fog.entities.FogDevice;

/**
 * CAR (Capability-aware Adaptive Route optimization) — region partitioning and
 * UAV capability assessment, following:
 *
 *   "UAV Trajectory Optimization Based on Pointer Networks and Adaptive Region
 *    Partitioning" (Guo, Tang, Tan, Luo, Zhao — IEEE IoT Journal).
 *
 * Stage 1 — UAV capability assessment (Eq. 15): a fixed synthetic region subset
 *   (size n*) is used to score every UAV from its attributes only; a LOWER score
 *   means a MORE capable UAV.
 *
 *   Scan width (Eq. 1):  R_i = 2 tan(alpha/2) * H / cos(beta)
 *   Scan time  (Eq. 7):  T^S_j_i = S_j / (R_i * v^s_i)
 *   Scan energy(Eq. 8):  p^S_j_i = q_i * T^S_j_i
 *   Flight time(Eq. 9):  T^U_ij,k = D_jk / v_i
 *   Flight ener(Eq. 10): p^U_ij,k = q_i * T^U_ij,k
 *   Capability (Eq. 15): F(Ui) = d*SUM(T^S + p^S) + (1-d)*SUM(T^U + p^U), d = 0.5
 *
 * Stage 2 — density/distance adaptive clustering (Eqs. 16-17):
 *   Local density:  phi_j = SUM_k chi(D_jk, dc) * (T^S_k_i + T^S_j_i + lam*T^U_ij,k)
 *   chi = 1 if D_jk <= dc else 0;  lam = 1
 *   dmin_j = min(D_jk) over k with phi_k > phi_j
 *   omega_j = phi_j * dmin_j
 *   Cluster centers = regions ranked above the pronounced "jump" in the
 *   descending omega sequence, detected automatically with Otsu adaptive
 *   thresholding on the adjacent-difference series.
 *   Remaining regions assigned to the nearest center; then clusters are balanced
 *   to the UAV count so every UAV owns exactly one cluster.
 *
 *   Sequential matching: higher-capability UAV (lower F) is assigned to the
 *   higher-omega (larger / more complex) cluster.
 */
public class CARRegionClustering {

    // Region scan area ranges kept in [50, 100] m^2 like the paper (Fig. 3 caption).
    private static final double MIN_AREA = 50.0;
    private static final double AREA_SPAN = 50.0;

    // d (Eq. 15): weight balancing scanning vs flight performance (paper sets 0.5).
    private static final double DELTA = 0.5;

    // lambda (Eq. 16): relative importance of flight time vs scanning time (paper sets 1).
    private static final double LAMBDA = 1.0;

    // mu: system resistance factor affecting scanning speed; v^s_i = mu * v_i (paper Sec. V-A).
    private static final double MU = 1.0;

    // Size of the fixed region subset used for capability assessment (n* < n).
    private static final int N_STAR = 10;

    // Maximum flight speed used to give each UAV a deterministic heterogeneous profile.
    private static final double SPEED_BASE = 12.0;
    private static final double SPEED_STEP = 2.0;
    private static final double POWER_BASE = 0.8;
    private static final double POWER_STEP = 0.1;

    /** The CAR attributes assigned to a UAV. */
    public static class UavProfile {
        public final double speedVm;        // v_i — average flight speed (m/s)
        public final double scanSpeedVs;    // v^s_i = mu * v_i
        public final double scanWidthR;     // R_i
        public final double powerQ;         // q_i — average power consumption
        public final double capabilityF;    // F(Ui) — Eq. 15 (lower is more capable)

        UavProfile(double speed, double scanWidth, double power, double capability) {
            this.speedVm = speed;
            this.scanSpeedVs = MU * speed;
            this.scanWidthR = scanWidth;
            this.powerQ = power;
            this.capabilityF = capability;
        }
    }

    /** A region (task area) used for clustering: location + scan area S_j. */
    public static class Region {
        public final int index;       // index into the MD list
        public final double x, y;     // region coordinates
        public final double area;     // S_j — region area to be scanned
        double phi = 0.0;             // local density (Eq. 16)
        double dmin = Double.MAX_VALUE; // min distance to a denser region
        double omega = 0.0;           // phi * dmin

        Region(int index, double x, double y, double area) {
            this.index = index;
            this.x = x;
            this.y = y;
            this.area = area;
        }
    }

    /** A cluster of regions assigned to a single UAV, with a complexity score. */
    public static class Cluster {
        public final double omega;             // complexity proxy of the cluster
        public final int centerRegion;         // center region index (region array index)
        public final List<Integer> regionIndices = new ArrayList<>();

        Cluster(double omega, int centerRegion) {
            this.omega = omega;
            this.centerRegion = centerRegion;
        }
    }

    /** One planned assignment: which waypoint order a UAV should follow. */
    public static class Plan {
        public final FogDevice uav;
        public final double[] wayXs;
        public final double[] wayYs;

        Plan(FogDevice uav, List<Integer> regionIndices, List<FogDevice> mds) {
            this.uav = uav;
            this.wayXs = new double[regionIndices.size()];
            this.wayYs = new double[regionIndices.size()];
            for (int i = 0; i < regionIndices.size(); i++) {
                FogDevice md = mds.get(regionIndices.get(i));
                wayXs[i] = md.x_coord;
                wayYs[i] = md.y_coord;
            }
        }
    }

    private CARRegionClustering() {
    }

    /* ------------------------------------------------------------------ */
    /* Stage 1: UAV capability assessment (Eq. 15)                         */
    /* ------------------------------------------------------------------ */

    /**
     * Builds a CAR profile per UAV. The capability term F(Ui) is computed on a
     * FIXED synthetic subset of N_STAR regions (uniform grid of the 2x2 km area),
     * identical for every UAV, exactly as the paper keeps (n*, region subset)
     * fixed across UAVs for a consistent comparison.
     */
    public static List<UavProfile> assessUavCapabilities(List<FogDevice> uavs) {
        double[] aX = new double[N_STAR];
        double[] aY = new double[N_STAR];
        double[] aS = new double[N_STAR];
        int cols = (int) Math.ceil(Math.sqrt(N_STAR));
        for (int j = 0; j < N_STAR; j++) {
            aX[j] = ((j % cols) + 0.5) * (2000.0 / cols);
            aY[j] = ((j / cols) + 0.5) * (2000.0 / (int) Math.ceil(N_STAR / (double) cols));
            aS[j] = MIN_AREA + (j % AREA_SPAN);
        }

        List<UavProfile> profiles = new ArrayList<>();
        for (int i = 0; i < uavs.size(); i++) {
            FogDevice uav = uavs.get(i);
            double speed = SPEED_BASE + (i % 6) * SPEED_STEP;              // 12..22 m/s
            double power = POWER_BASE + (i % 5) * POWER_STEP;              // 0.8..1.2 W-ish
            double width = uav.coverageRadius > 0 ? uav.coverageRadius : 50.0;

            double scanCost = 0.0;
            double flightCost = 0.0;
            for (int j = 0; j < N_STAR; j++) {
                double scanTime = aS[j] / (width * (MU * speed));          // Eq. 7
                double scanEnergy = power * scanTime;                      // Eq. 8
                scanCost += scanTime + scanEnergy;
                for (int k = 0; k < N_STAR; k++) {
                    if (j == k) continue;
                    double d = distance(aX[j], aY[j], aX[k], aY[k]);
                    double flightTime = d / speed;                         // Eq. 9
                    double flightEnergy = power * flightTime;              // Eq. 10
                    flightCost += flightTime + flightEnergy;
                }
            }
            double capability = DELTA * scanCost + (1.0 - DELTA) * flightCost; // Eq. 15
            profiles.add(new UavProfile(speed, width, power, capability));
        }
        return profiles;
    }

    /* ------------------------------------------------------------------ */
    /* Stage 2: region partitioning (Eqs. 16-17) + Otsu centers             */
    /* ------------------------------------------------------------------ */

    /**
     * Runs the adaptive clustering over the given ground devices (MDs).
     * Returns one Cluster per UAV ordered by descending complexity (omega),
     * balanced so that every UAV owns a cluster.
     */
    public static List<Cluster> partitionRegions(List<FogDevice> mds, List<FogDevice> uavs,
                                                 List<UavProfile> profiles) {
        int n = mds.size();
        if (n == 0) return new ArrayList<>();

        // Build regions (index, position, scan area S_j).
        List<Region> regions = new ArrayList<>();
        for (int j = 0; j < n; j++) {
            FogDevice md = mds.get(j);
            double area = MIN_AREA + (j % AREA_SPAN);
            regions.add(new Region(j, md.x_coord, md.y_coord, area));
        }

        // Cut-off distance dc: derived from the interregional-distance distribution.
        double dc = cutoffDistance(regions);

        // Pairwise distances.
        double[][] dist = new double[n][n];
        for (int j = 0; j < n; j++) {
            for (int k = j + 1; k < n; k++) {
                dist[j][k] = dist[k][j] = distance(regions.get(j), regions.get(k));
            }
        }

        // Local density phi_j (Eq. 16), averaged over all UAV profiles (org-fog sim
        // treats regions w.r.t. an "average" UAV, so use the mean flight/scan times).
        double avgScanTime = 0.0, avgFlightTime = 0.0;
        for (UavProfile p : profiles) {
            double area = MIN_AREA + AREA_SPAN / 2.0;
            avgScanTime += area / (p.scanWidthR * p.scanSpeedVs);
            double dAv = avgPairwiseDist(regions);
            avgFlightTime += dAv / p.speedVm;
        }
        avgScanTime /= Math.max(1, profiles.size());
        avgFlightTime /= Math.max(1, profiles.size());

        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                boolean inRange = (j != k) && dist[j][k] <= dc;
                if (inRange) {
                    // chi(D_jk, dc) * (T^S_k + T^S_j + lam * T^U_jk)
                    sum += avgScanTime + avgScanTime + LAMBDA * dist[j][k] / avgSpeed(profiles);
                }
            }
            regions.get(j).phi = sum;
        }

        // dmin_j = min flight distance to any region with higher density.
        for (int j = 0; j < n; j++) {
            double dmin = Double.MAX_VALUE;
            for (int k = 0; k < n; k++) {
                if (regions.get(k).phi > regions.get(j).phi) {
                    dmin = Math.min(dmin, dist[j][k]);
                }
            }
            regions.get(j).dmin = (dmin == Double.MAX_VALUE) ? maxDist(dist) : dmin;
        }

        // omega_j = phi_j * dmin_j.
        for (Region r : regions) r.omega = r.phi * r.dmin;

        // Order by descending omega. Cluster centers = regions above the Otsu-detected
        // "jump point" in the adjacent-difference sequence (paper Sec. IV-B).
        Region[] sorted = regions.toArray(new Region[0]);
        Arrays.sort(sorted, new Comparator<Region>() {
            public int compare(Region a, Region b) {
                return Double.compare(b.omega, a.omega);
            }
        });

        int nCenters = detectCenterCount(sorted);

        // Build initial clusters around each center.
        List<Cluster> clusters = new ArrayList<>();
        boolean[] isCenter = new boolean[n];
        for (int c = 0; c < nCenters; c++) {
            isCenter[sorted[c].index] = true;
            clusters.add(new Cluster(sorted[c].omega, sorted[c].index));
        }
        if (clusters.isEmpty()) {
            clusters.add(new Cluster(1.0, sorted[0].index));
            isCenter[sorted[0].index] = true;
        }

        // Assign each remaining region to the nearest center region.
        for (Region r : regions) {
            if (isCenter[r.index]) continue;
            int bestC = 0;
            double bestD = Double.MAX_VALUE;
            for (int c = 0; c < clusters.size(); c++) {
                double d = distance(r, regions.get(clusters.get(c).centerRegion));
                if (d < bestD) {
                    bestD = d;
                    bestC = c;
                }
            }
            clusters.get(bestC).regionIndices.add(r.index);
        }
        // The center regions themselves.
        for (int c = 0; c < nCenters && c < clusters.size(); c++) {
            clusters.get(c).regionIndices.add(sorted[c].index);
        }

        // Balance clusters to the UAV count so each UAV owns exactly one cluster.
        clusters = balanceClusters(clusters, uavs.size(), regions, mds);

        // Order clusters by descending complexity (omega) for capability matching.
        clusters.sort(new Comparator<Cluster>() {
            public int compare(Cluster a, Cluster b) {
                return Double.compare(b.omega, a.omega);
            }
        });
        return clusters;
    }

    private static double cutoffDistance(List<Region> regions) {
        int n = regions.size();
        if (n < 2) return 100.0;
        double[] ds = new double[n * (n - 1) / 2];
        int idx = 0;
        for (int j = 0; j < n; j++) {
            for (int k = j + 1; k < n; k++) {
                ds[idx++] = distance(regions.get(j), regions.get(k));
            }
        }
        Arrays.sort(ds);
        return ds[(int) (0.25 * (ds.length - 1))]; // lower quartile of pairwise distances
    }

    private static double avgPairwiseDist(List<Region> regions) {
        int n = regions.size();
        if (n < 2) return 100.0;
        double sum = 0.0;
        int count = 0;
        for (int j = 0; j < n; j++) {
            for (int k = j + 1; k < n; k++) {
                sum += distance(regions.get(j), regions.get(k));
                count++;
            }
        }
        return sum / count;
    }

    private static double avgSpeed(List<UavProfile> profiles) {
        if (profiles.isEmpty()) return 12.0;
        double sum = 0.0;
        for (UavProfile p : profiles) sum += p.speedVm;
        return sum / profiles.size();
    }

    private static double maxDist(double[][] dist) {
        double m = 100.0;
        for (double[] row : dist) {
            for (double d : row) m = Math.max(m, d);
        }
        return m;
    }

    /**
     * Detects the number of cluster centers using Otsu adaptive thresholding on
     * the adjacent-difference sequence of the (descending) omega ranking.
     * The pronounced jump separates centers (above) from non-centers (below).
     */
    private static int detectCenterCount(Region[] ranked) {
        int n = ranked.length;
        if (n <= 2) return 1;
        double[] dif = new double[n - 1];
        int maxIdx = 0;
        for (int i = 0; i < n - 1; i++) {
            dif[i] = ranked[i].omega - ranked[i + 1].omega;
            if (dif[i] > dif[maxIdx]) maxIdx = i;
        }
        double thr = otsuThreshold(dif);
        // The biggest gap must be statistically significant; otherwise one cluster.
        if (dif[maxIdx] <= thr) return 1;
        return maxIdx + 1;
    }

    /** Otsu adaptive threshold over an array of non-negative values. */
    private static double otsuThreshold(double[] values) {
        double maxV = 0.0;
        for (double v : values) maxV = Math.max(maxV, v);
        if (maxV <= 0.0) return 0.0;

        final int bins = 64;
        long[] hist = new long[bins];
        for (double v : values) {
            int b = (int) (v / maxV * (bins - 1));
            hist[b]++;
        }
        long total = values.length;
        double sum = 0.0;
        for (int b = 0; b < bins; b++) sum += b * hist[b];

        double bestVar = -1.0;
        double bestThr = 0.0;
        long wB = 0;
        double sumB = 0.0;
        for (int b = 0; b < bins; b++) {
            wB += hist[b];
            if (wB == 0) continue;
            long wF = total - wB;
            if (wF == 0) break;
            sumB += b * hist[b];
            double mB = sumB / wB;
            double mF = (sum - sumB) / wF;
            double betweenVar = (double) wB * wF * (mB - mF) * (mB - mF);
            if (betweenVar > bestVar) {
                bestVar = betweenVar;
                bestThr = ((b + 0.5) / bins) * maxV;
            }
        }
        return bestThr;
    }

    private static List<Cluster> balanceClusters(List<Cluster> clusters, int numUavs,
                                                 List<Region> regions, List<FogDevice> mds) {
        if (clusters.isEmpty()) return clusters;
        if (clusters.size() > numUavs) {
            // Keep only the numUavs most complex clusters; reassign members to nearest kept center.
            List<Cluster> kept = new ArrayList<>(clusters.subList(0, numUavs));
            List<Integer> allRegions = new ArrayList<>();
            for (Cluster c : clusters) allRegions.addAll(c.regionIndices);
            for (Cluster c : kept) c.regionIndices.clear();
            for (int regIdx : allRegions) {
                int bestC = 0;
                double bestD = Double.MAX_VALUE;
                for (int c = 0; c < kept.size(); c++) {
                    Region center = regions.get(kept.get(c).centerRegion);
                    double d = distance(regions.get(regIdx).x, regions.get(regIdx).y, center.x, center.y);
                    if (d < bestD) {
                        bestD = d;
                        bestC = c;
                    }
                }
                kept.get(bestC).regionIndices.add(regIdx);
            }
            return kept;
        }
        // Split the largest clusters until we have numUavs of them.
        while (clusters.size() < numUavs) {
            Cluster largest = null;
            for (Cluster c : clusters) {
                if (largest == null || c.regionIndices.size() > largest.regionIndices.size()) largest = c;
            }
            if (largest == null || largest.regionIndices.size() < 2) break;
            splitCluster(largest, clusters, regions);
        }
        return clusters;
    }

    /** Deterministic 2-medoid split of a cluster along its farthest pair. */
    private static void splitCluster(Cluster src, List<Cluster> clusters, List<Region> regions) {
        List<Integer> members = src.regionIndices;
        Region a = regions.get(members.get(0));
        Region b = null;
        double bestD = -1.0;
        for (int idx : members) {
            Region r = regions.get(idx);
            double d = distance(a, r);
            if (d > bestD) {
                bestD = d;
                b = r;
            }
        }
        if (b == null || b == a) return;

        Cluster cA = new Cluster(src.omega, a.index);
        Cluster cB = new Cluster(src.omega, b.index);
        for (int idx : members) {
            Region r = regions.get(idx);
            double dA = distance(r, a);
            double dB = distance(r, b);
            (dA <= dB ? cA : cB).regionIndices.add(idx);
        }
        if (cA.regionIndices.isEmpty() || cB.regionIndices.isEmpty()) return; // unchanged

        clusters.remove(src);
        clusters.add(cA);
        clusters.add(cB);
    }

    /* ------------------------------------------------------------------ */
    /* Sequential matching (paper Sec. IV-B)                               */
    /* ------------------------------------------------------------------ */

    /**
     * Matches UAVs (by capability, ascending F) to clusters (by descending omega).
     * Returns a Plan for each UAV with its cluster's waypoint order (raw, unplanned —
     * ordering is done by the pointer network afterwards).
     */
    public static List<Plan> matchAndPlan(List<FogDevice> uavs, List<FogDevice> mds,
                                          List<UavProfile> profiles, List<Cluster> clusters) {
        List<Plan> plans = new ArrayList<>();
        for (int i = 0; i < uavs.size(); i++) {
            // The i-th most capable UAV gets the i-th most complex cluster.
            FogDevice uav = uavs.get(rankByCapability(profiles, i));
            Cluster cluster = clusters.size() > i ? clusters.get(i) : clusters.get(clusters.size() - 1);
            plans.add(new Plan(uav, cluster.regionIndices, mds));
        }
        return plans;
    }

    /** Index of the UAV with the k-th lowest capability score (0 = most capable). */
    private static int rankByCapability(List<UavProfile> profiles, int k) {
        Integer[] idx = new Integer[profiles.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, Comparator.comparingDouble((Integer i) -> profiles.get(i).capabilityF));
        return idx[Math.min(k, idx.length - 1)];
    }

    /* ------------------------------------------------------------------ */
    /* Geometry                                                            */
    /* ------------------------------------------------------------------ */

    private static double distance(Region a, Region b) {
        return distance(a.x, a.y, b.x, b.y);
    }

    private static double distance(double x1, double y1, double x2, double y2) {
        double dx = x2 - x1;
        double dy = y2 - y1;
        return Math.sqrt(dx * dx + dy * dy);
    }
}