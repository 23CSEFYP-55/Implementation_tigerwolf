package org.fog.test.perfeval;

import java.util.ArrayList;
import java.util.Calendar;
import java.util.LinkedList;
import java.util.List;

import org.cloudbus.cloudsim.Host;
import org.cloudbus.cloudsim.Log;
import org.cloudbus.cloudsim.Pe;
import org.cloudbus.cloudsim.Storage;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.power.PowerHost;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;

import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.Application;
import org.fog.application.selectivity.FractionalSelectivity;
import org.fog.entities.Actuator;
import org.fog.entities.FogBroker;
import org.fog.entities.FogDevice;
import org.fog.entities.FogDeviceCharacteristics;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.placement.Controller;
import org.fog.placement.ModuleMapping;
import org.fog.placement.ModulePlacementEdgewards;
import org.fog.placement.ModulePlacementMOGS;
import org.fog.fanet.DynamicRepairScheduler;
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;
import org.fog.utils.distribution.PoissonDistribution;
import org.fog.utils.distribution.UniformDistribution;
public class FANETSimulation {

    // All fog devices and sensors/actuators in the simulation
    static List<FogDevice> fogDevices = new ArrayList<>();
    static List<Sensor> sensors = new ArrayList<>();
    static List<Actuator> actuators = new ArrayList<>();
    static Controller masterController;

    // Application, UAV count, and HARD LIMITS
    static int numUAVs = 30;
    static int NUM_MDS = 100; // 100 MDs
    static double SENSOR_TRANSMISSION_TIME = 1000.0; // Mean time (lambda) for Poisson

    // --- NEW: Time Limit ---
    static double MAX_SIM_TIME = 5000.0;

    public static void main(String[] args) {
        String schedulerType = "DYNAMIC"; // Default
        if (args.length > 0) {
            schedulerType = args[0];
        }
        
        String trajectoryType = "PPO"; // Default
        if (args.length > 1) {
            trajectoryType = args[1];
        }
        
        Log.printLine("========== Starting FANET Simulation with " + schedulerType
                + " Scheduler / " + trajectoryType + " Trajectory ==========");

        try {
            // 1. Initialize CloudSim
            int numUsers = 1;
            Calendar calendar = Calendar.getInstance();
            boolean traceFlag = false;
            CloudSim.init(numUsers, calendar, traceFlag);

            // 2. Create the Application definition
            String appId = "fanet_app";
            FogBroker broker = new FogBroker("broker");
            Application application = createApplication(appId, broker.getId());
            application.setUserId(broker.getId());

            // Dynamic UAV count between 2 and 30
            numUAVs = 30;
            System.out.println("Dynamically spawning " + numUAVs + " UAVs...");

            Log.disable(); // Directive 1: Silence Default Logging (Noise Reduction)

            // 3. Create the physical topology (UAVs + Ground Devices)
            createFogDevices(broker.getId(), appId);

            // 4. Create the Controller (which now has our task scheduling logic)
            masterController = new Controller("master-controller", fogDevices, sensors, actuators);
            if (schedulerType.equalsIgnoreCase("MOGS")) {
                masterController.setTaskScheduler(new org.fog.fanet.MOGSScheduler());
            } else if (schedulerType.equalsIgnoreCase("LEXICOGRAPHIC")) {
                masterController.setTaskScheduler(new org.fog.fanet.LexicographicScheduler());
            } else if (schedulerType.equalsIgnoreCase("BIDDING")) {
                masterController.setTaskScheduler(new org.fog.fanet.BiddingScheduler());
            } else if (schedulerType.equalsIgnoreCase("MARL")) {
                masterController.setTaskScheduler(new org.fog.fanet.MARLScheduler());
            } else {
                masterController.setTaskScheduler(new org.fog.fanet.DynamicRepairScheduler());
            }

            // 5. Set the module placement policy (EdgeWards = prefer edge/UAV nodes)
            ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();
            for (int i = 0; i < NUM_MDS; i++) {
                moduleMapping.addModuleToDevice("sensor_module", "ground_device_" + i);
            }
            masterController.submitApplication(application,
                    new ModulePlacementMOGS(fogDevices, sensors, actuators, application, moduleMapping));

            // 6. Record simulation start time and run
            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());

            // Connect to the Trajectory AI Server
            org.fog.fanet.TrajectoryModel trajectoryModel;
            if (trajectoryType.equalsIgnoreCase("CAR")) {
                trajectoryModel = new org.fog.fanet.CarTrajectoryModel();
            } else {
                trajectoryModel = new org.fog.fanet.PPOTrajectoryModel();
            }
            trajectoryModel.start();
            masterController.setTrajectoryModel(trajectoryModel);

            org.fog.fanet.MarlClient.connect();

            // Enforce simulation time
            System.out.println("Scheduling simulation termination at " + MAX_SIM_TIME + " ms.");
            CloudSim.terminateSimulation(MAX_SIM_TIME);

            // Start the simulation loop
            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            // --- CRITICAL STEP FOR SHUTDOWN ---
            trajectoryModel.stop();
            org.fog.fanet.MarlClient.disconnect();

            Log.enable(); // Re-enable logging for the final dashboard

            // Directive 3: Generate the Final Benchmark Dashboard
            printUAVBenchmarks();
            printSchedulingMetrics();
            printMogsBenchDashboard();

            System.out.println("========== FANET Simulation Finished ==========");

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("An error occurred during the FANET simulation.");
        }
    }

    private static void printSchedulingMetrics() {
        if (masterController == null) return;
        System.out.println("========== SCHEDULING ALGORITHM METRICS ==========");
        long scheduled = masterController.systemTotalScheduled;
        long dropped = masterController.systemTotalDropped;
        long total = scheduled + dropped;
        double successRate = total > 0 ? ((double) scheduled / total) * 100.0 : 0.0;
        
        long totalOffloaded = 0;
        for (FogDevice device : fogDevices) {
            if (device.getName().startsWith("uav")) {
                totalOffloaded += device.tasksOffloaded;
            }
        }
        
        double avgOverheadMs = masterController.schedulingInvocations > 0 
                ? (masterController.totalSchedulingTimeNs / (double) masterController.schedulingInvocations) / 1_000_000.0 
                : 0.0;

        System.out.println(String.format("Total Tasks Successfully Scheduled : %d", scheduled));
        System.out.println(String.format("Total Tasks Dropped              : %d", dropped));
        System.out.println(String.format("Scheduling Success Rate          : %.2f%%", successRate));
        System.out.println(String.format("Average Scheduling Overhead      : %.4f ms per batch", avgOverheadMs));
        System.out.println(String.format("System Total Offloaded Tasks     : %d", totalOffloaded));
        System.out.println("===================================================\n");
    }

    private static void printMogsBenchDashboard() {
        if (masterController == null) return;
        
        long invocations = masterController.schedulingInvocations;
        if (invocations == 0) invocations = 1; // Avoid division by zero
        
        long totalCompleted = 0;
        for (FogDevice device : fogDevices) {
            if (device.getName().startsWith("uav")) {
                totalCompleted += device.completedTasks;
            }
        }
        
        long totalFailed = masterController.systemTotalDropped;
        long totalGenerated = masterController.totalGeneratedTasks;
        long totalUnassigned = masterController.totalUnassignedTasks;
        
        double avgThroughput = totalGenerated > 0 ? (double) totalCompleted / totalGenerated : 0.0;
        double avgRuntimeSec = (masterController.totalSchedulingTimeNs / (double) invocations) / 1_000_000_000.0;
        double avgUavUtil = masterController.totalUAVUtilization / invocations;
        double avgCapUtil = masterController.totalCapacityUtilization / invocations;
        
        System.out.println("\n               UAV Task Scheduling Benchmark Results               ");
        System.out.println("╭──────────────────────────┬─────────────────────╮");
        System.out.println("│ Metric                   │ Value               │");
        System.out.println("├──────────────────────────┼─────────────────────┤");
        System.out.println(String.format("│ Average Throughput       │ %-19.4f │", avgThroughput));
        System.out.println(String.format("│ Average Runtime (sec)    │ %-19.6f │", avgRuntimeSec));
        System.out.println(String.format("│ Average UAV Utilization  │ %-19.4f │", avgUavUtil));
        System.out.println(String.format("│ Average Capacity Util    │ %-19.4f │", avgCapUtil));
        System.out.println(String.format("│ Average Completed Tasks  │ %-19.2f │", (double) totalCompleted / invocations));
        System.out.println(String.format("│ Average Failed Tasks     │ %-19.2f │", (double) totalFailed / invocations));
        System.out.println(String.format("│ Average Unassigned Tasks │ %-19.2f │", (double) totalUnassigned / invocations));
        
        // Extract TaskScheduler to print Dynamic Repair specific metrics
        try {
            java.lang.reflect.Field field = org.fog.placement.Controller.class.getDeclaredField("taskScheduler");
            field.setAccessible(true);
            Object scheduler = field.get(masterController);
            if (scheduler instanceof org.fog.fanet.DynamicRepairScheduler) {
                org.fog.fanet.DynamicRepairScheduler drs = (org.fog.fanet.DynamicRepairScheduler) scheduler;
                long drsCycles = drs.getSchedulingCycles() > 0 ? drs.getSchedulingCycles() : 1;
                System.out.println(String.format("│ Average Repair Queue     │ %-19.2f │", (double) drs.getTotalRepairQueueLength() / drsCycles));
                System.out.println(String.format("│ Average Invalid Assigns  │ %-19.2f │", (double) drs.getTotalInvalidAssignments() / drsCycles));
                System.out.println(String.format("│ Average Reassigned Tasks │ %-19.2f │", (double) drs.getTotalReassignedTasks() / drsCycles));
            }
        } catch (Exception e) {
            // Ignore reflection errors if any
        }
        
        System.out.println("╰──────────────────────────┴─────────────────────╯\n");
    }

    private static void printUAVBenchmarks() {
        System.out.println("\n========== FINAL UAV BENCHMARK DASHBOARD ==========");
        System.out.println(String.format("%-15s | %-15s | %-20s | %-16s | %-19s | %-15s", 
                "UAV ID", "Tasks Completed", "Data Processed (MB)", "Avg Latency (ms)", "Energy Consumed (J)", "Tasks Offloaded"));
        System.out.println("----------------------------------------------------------------------------------------------------------------------");

        long totalSystemTasks = 0;
        double totalSystemData = 0.0;
        double totalSystemLatency = 0.0;

        for (FogDevice device : fogDevices) {
            if (device.getName().startsWith("uav")) {
                int completed = device.completedTasks;
                double dataMB = device.totalDataProcessed / (1024.0 * 1024.0);
                double avgLatency = (completed > 0) ? (device.totalLatency / completed) : 0.0;
                double energy = device.getEnergyConsumption();
                int offloaded = device.tasksOffloaded;

                totalSystemTasks += completed;
                totalSystemData += dataMB;
                totalSystemLatency += device.totalLatency;

                System.out.println(String.format("%-15s | %-15d | %-20.4f | %-16.2f | %-19.4f | %-15d", 
                        device.getName(), completed, dataMB, avgLatency, energy, offloaded));
            }
        }

        System.out.println("----------------------------------------------------------------------------------------------------------------------");
        double systemAvgLatency = (totalSystemTasks > 0) ? (totalSystemLatency / totalSystemTasks) : 0.0;
        System.out.println(String.format("System Total Tasks Completed: %d  |  System Total Throughput: %.4f MB  |  Average System Latency: %.2f ms", totalSystemTasks, totalSystemData, systemAvgLatency));
        System.out.println("===================================================\n");
    }

    /**
     * Creates the physical network topology:
     * - 1 Cloud node (top level)
     * - 3 UAV fog devices (named "uav_0", "uav_1", "uav_2") — these are picked up by the scheduler
     * - 1 Ground sensor device (mobile end device)
     */
    private static void createFogDevices(int userId, String appId) {
        java.util.Random rand = new java.util.Random();

        // --- UAV NODES (Edge Layer) ---
        // These are named "uav_X" so the task scheduler identifies them
        for (int i = 0; i < numUAVs; i++) {
            long mips = 2800 + (i * 800); // Give each drone a different MIPS capacity
            FogDevice uav = createFogDevice("uav_" + i, mips, 4000, 10000, 1000, 1,
                    0.0, 107.339, 83.4333);
            uav.setParentId(-1); // No cloud, so no parent
            uav.setUplinkLatency(5);

            // Set FANET-specific properties (coordinates and task capacity)
            uav.x_coord = rand.nextDouble() * 2000.0;
            uav.y_coord = rand.nextDouble() * 2000.0;
            uav.taskCapacity = 150;
            uav.coverageRadius = 20.0 + (rand.nextDouble() * 120.0); // 20m to 140m

            fogDevices.add(uav);
        }

        // --- 800 GROUND SENSOR DEVICES (MDs) ---
        for (int i = 0; i < NUM_MDS; i++) {
            FogDevice groundDevice = createFogDevice("ground_device_" + i, 1000, 1000, 10000, 270,
                    3, 0.0, 87.53, 82.44);
            // Standalone devices, task scheduler handles routing
            groundDevice.setParentId(-1); 
            groundDevice.x_coord = rand.nextDouble() * 2000.0;
            groundDevice.y_coord = rand.nextDouble() * 2000.0;
            fogDevices.add(groundDevice);

            // --- SENSOR (generates tasks via Poisson Process) ---
            Sensor sensor = new Sensor("sensor_" + i, "SENSOR_DATA", userId, appId,
                    new PoissonDistribution(SENSOR_TRANSMISSION_TIME));
            sensor.setGatewayDeviceId(groundDevice.getId());
            sensor.setTransmissionStartDelay(rand.nextInt(1000));
            sensor.setLatency(1.0);
            sensors.add(sensor);

            // --- ACTUATOR (receives results) ---
            Actuator actuator = new Actuator("actuator_" + i, userId, appId, "ACTUATOR_CMD");
            actuator.setGatewayDeviceId(groundDevice.getId());
            actuator.setLatency(1.0);
            actuators.add(actuator);
        }
    }

    /**
     * Defines the application: what tasks look like and how they flow
     */
    @SuppressWarnings({"serial"})
    private static Application createApplication(String appId, int userId) {

        Application application = Application.createApplication(appId, userId);

        // Define app modules (processing stages)
        application.addAppModule("sensor_module", 10);       // runs on ground device
        application.addAppModule("processing_module", 10);   // runs on UAV (edge)

        // Define data flow edges between modules
        application.addAppEdge("SENSOR_DATA", "sensor_module", 100, 500000,
                "SENSOR_DATA", Tuple.UP, AppEdge.SENSOR);

        // sensor_module → processing_module: preprocessed data
        application.addAppEdge("sensor_module", "processing_module", 100, 500000,
                "PREPROCESSED_DATA", Tuple.UP, AppEdge.MODULE);

        // processing_module → actuator: command output
        application.addAppEdge("processing_module", "ACTUATOR_CMD", 100, 500000,
                "ACTUATOR_CMD", Tuple.DOWN, AppEdge.ACTUATOR);

        // Define selectivity (how many output tuples per input tuple)
        application.addTupleMapping("sensor_module", "SENSOR_DATA", "PREPROCESSED_DATA",
                new FractionalSelectivity(1.0));
        application.addTupleMapping("processing_module", "PREPROCESSED_DATA", "ACTUATOR_CMD",
                new FractionalSelectivity(1.0));

        // Define the end-to-end loop for latency measurement
        final AppLoop loop = new AppLoop(new ArrayList<String>() {{
            add("SENSOR_DATA");
            add("sensor_module");
            add("processing_module");
            add("ACTUATOR_CMD");
        }});
        List<AppLoop> loops = new ArrayList<>();
        loops.add(loop);
        application.setLoops(loops);

        return application;
    }

    /**
     * Helper method to create a FogDevice with standard parameters
     */
    private static FogDevice createFogDevice(String nodeName, long mips, int ram,
                                             long upBw, long downBw, int level, double ratePerMips,
                                             double busyPower, double idlePower) {

        List<Pe> peList = new ArrayList<>();
        peList.add(new Pe(0, new PeProvisionerSimple(mips)));

        int hostId = FogUtils.generateEntityId();
        long storage = 1000000;
        int bw = 10000;

        PowerHost host = new PowerHost(hostId, new RamProvisionerSimple(ram),
                new BwProvisionerSimple(bw), storage, peList,
                new StreamOperatorScheduler(peList),
                new FogLinearPowerModel(busyPower, idlePower));

        List<Host> hostList = new ArrayList<>();
        hostList.add(host);

        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double time_zone = 10.0;
        double cost = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;

        FogDeviceCharacteristics characteristics = new FogDeviceCharacteristics(
                arch, os, vmm, host, time_zone, cost, costPerMem,
                costPerStorage, costPerBw);

        FogDevice fogDevice = null;
        try {
            fogDevice = new FogDevice(nodeName, characteristics,
                    new AppModuleAllocationPolicy(hostList),
                    new LinkedList<Storage>(), 10, upBw, downBw, 0, ratePerMips);
        } catch (Exception e) {
            e.printStackTrace();
        }

        fogDevice.setLevel(level);
        return fogDevice;
    }
}