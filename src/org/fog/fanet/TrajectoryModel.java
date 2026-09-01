package org.fog.fanet;

import java.util.List;
import org.fog.entities.FogDevice;

public interface TrajectoryModel {
    /**
     * Start and initialize the trajectory model (e.g., connect to sockets).
     */
    void start();

    /**
     * Compute and apply the new coordinates for all UAVs.
     */
    void updateTrajectories(List<FogDevice> uavs, List<FogDevice> mds);

    /**
     * Stop and teardown the trajectory model (e.g., close sockets).
     */
    void stop();
}
