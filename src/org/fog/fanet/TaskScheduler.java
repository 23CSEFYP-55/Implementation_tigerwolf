package org.fog.fanet;

import org.fog.entities.FogDevice;
import org.fog.entities.Tuple;

import java.util.List;

public interface TaskScheduler {
    /**
     * Schedules tasks by assigning them to suitable UAVs.
     * 
     * @param tasks List of tasks to be scheduled.
     * @param uavs List of available UAVs (FogDevices).
     */
    void scheduleTasks(List<Tuple> tasks, List<FogDevice> uavs);
}
