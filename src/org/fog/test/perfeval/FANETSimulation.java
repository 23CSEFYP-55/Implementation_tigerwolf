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
import org.fog.policy.AppModuleAllocationPolicy;
import org.fog.scheduler.StreamOperatorScheduler;
import org.fog.utils.FogLinearPowerModel;
import org.fog.utils.FogUtils;
import org.fog.utils.TimeKeeper;
import org.fog.utils.distribution.DeterministicDistribution;
import org.fog.utils.distribution.UniformDistribution;

public class FANETSimulation {

    // All fog devices and sensors/actuators in the simulation
    static List<FogDevice> fogDevices = new ArrayList<>();
    static List<Sensor> sensors = new ArrayList<>();
    static List<Actuator> actuators = new ArrayList<>();

    // Application, UAV count, and HARD LIMITS
    static int numUAVs = 3;
    static double SENSOR_TRANSMISSION_TIME = 10.0; // ms between task emissions

    // --- NEW: Task Limit ---
    static int MAX_TASKS = 500;

    public static void main(String[] args) {
        Log.printLine("========== Starting FANET Simulation with MOGS Matching ==========");

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

            // 3. Create the physical topology (Cloud + UAVs + Ground Devices)
            createFogDevices(broker.getId(), appId);

            // 4. Create the Controller (which now has our MOGS logic)
            Controller controller = new Controller("master-controller", fogDevices, sensors, actuators);

            // 5. Set the module placement policy (EdgeWards = prefer edge/UAV nodes)
            ModuleMapping moduleMapping = ModuleMapping.createModuleMapping();
            moduleMapping.addModuleToDevice("storage_module", "cloud");
            controller.submitApplication(application,
                    new ModulePlacementMOGS(fogDevices, sensors, actuators, application, moduleMapping));


            // 6. Record simulation start time and run
            TimeKeeper.getInstance().setSimulationStartTime(Calendar.getInstance().getTimeInMillis());

            // Connect to the Python AI Server
            org.fog.fanet.PreferenceBuilder.connectToPPO();

            // --- NEW: Calculate exact stop time to enforce MAX_TASKS ---
            // If tasks emit every 10ms, stopping at 5000ms ensures exactly 500 tasks.
            double stopTime = MAX_TASKS * SENSOR_TRANSMISSION_TIME;
            Log.printLine("Scheduling simulation termination at " + stopTime + " ms to enforce task limit.");
            CloudSim.terminateSimulation(stopTime);

            // Start the simulation loop
            CloudSim.startSimulation();
            CloudSim.stopSimulation();

            // --- CRITICAL STEP FOR SHUTDOWN ---
            // Ensure that inside your disconnectFromPPO() method, you are sending
            // the `out.println("CLOSE");` command to the Python socket before closing it!
            org.fog.fanet.PreferenceBuilder.disconnectFromPPO();

            Log.printLine("========== FANET Simulation Finished ==========");

        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("An error occurred during the FANET simulation.");
        }
    }

    /**
     * Creates the physical network topology:
     * - 1 Cloud node (top level)
     * - 3 UAV fog devices (named "uav_0", "uav_1", "uav_2") — these are picked up by MOGS
     * - 1 Ground sensor device (mobile end device)
     */
    private static void createFogDevices(int userId, String appId) {

        // --- CLOUD NODE ---
        FogDevice cloud = createFogDevice("cloud", 44800, 40000, 100, 10000, 0,
                0.01, 16 * 103, 16 * 83.25);
        cloud.setParentId(-1); // Cloud has no parent
        fogDevices.add(cloud);

        // --- UAV NODES (Edge Layer) ---
        // These are named "uav_X" so the MOGS scheduler identifies them
        for (int i = 0; i < numUAVs; i++) {
            long mips = 2800 + (i * 800); // Give each drone a different MIPS capacity
            FogDevice uav = createFogDevice("uav_" + i, mips, 4000, 10000, 1000, 1,
                    0.0, 107.339, 83.4333);
            uav.setParentId(cloud.getId());
            uav.setUplinkLatency(5); // 5ms latency to cloud

            // Set FANET-specific properties (coordinates and task capacity)
            uav.x_coord = (i + 1) * 100.0; // Spread UAVs spatially
            uav.y_coord = (i + 1) * 50.0;
            uav.taskCapacity = 10;          // Each UAV handles up to 10 tasks

            fogDevices.add(uav);
        }

        // --- GROUND SENSOR DEVICE ---
        FogDevice groundDevice = createFogDevice("ground_device_0", 1000, 1000, 10000, 270,
                3, 0.0, 87.53, 82.44);
        // Connect ground device to uav_0 as its parent
        groundDevice.setParentId(fogDevices.get(1).getId()); // uav_0
        groundDevice.setUplinkLatency(2);
        fogDevices.add(groundDevice);

        // --- SENSOR (generates the actual tasks/tuples) ---
        Sensor sensor = new Sensor("sensor_0", "SENSOR_DATA", userId, appId,
                new UniformDistribution(SENSOR_TRANSMISSION_TIME - 3.0, SENSOR_TRANSMISSION_TIME + 3.0));
        sensor.setGatewayDeviceId(groundDevice.getId());
        sensor.setLatency(1.0); // 1ms sensor-to-device latency
        sensors.add(sensor);

        // --- ACTUATOR (receives results) ---
        Actuator actuator = new Actuator("actuator_0", userId, appId, "ACTUATOR_CMD");
        actuator.setGatewayDeviceId(groundDevice.getId());
        actuator.setLatency(1.0);
        actuators.add(actuator);
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
        application.addAppModule("storage_module", 10);      // runs on cloud

        // Define data flow edges between modules
        // Sensor → sensor_module: raw sensor data (1000 bits, 500 MI to process)
        application.addAppEdge("SENSOR_DATA", "sensor_module", 1000, 500,
                "SENSOR_DATA", Tuple.UP, AppEdge.SENSOR);

        // sensor_module → processing_module: preprocessed data
        application.addAppEdge("sensor_module", "processing_module", 2000, 1000,
                "PREPROCESSED_DATA", Tuple.UP, AppEdge.MODULE);

        // processing_module → storage_module: processed result
        application.addAppEdge("processing_module", "storage_module", 500, 200,
                "RESULT_DATA", Tuple.UP, AppEdge.MODULE);

        // processing_module → actuator: command output
        application.addAppEdge("processing_module", "ACTUATOR_CMD", 100, 50,
                "ACTUATOR_CMD", Tuple.DOWN, AppEdge.ACTUATOR);

        // Define selectivity (how many output tuples per input tuple)
        application.addTupleMapping("sensor_module", "SENSOR_DATA", "PREPROCESSED_DATA",
                new FractionalSelectivity(1.0));
        application.addTupleMapping("processing_module", "PREPROCESSED_DATA", "RESULT_DATA",
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