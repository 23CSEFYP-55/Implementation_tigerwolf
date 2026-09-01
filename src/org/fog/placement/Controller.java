package org.fog.placement;

import java.util.Calendar;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.SimEntity;
import org.cloudbus.cloudsim.core.SimEvent;
import org.fog.application.AppEdge;
import org.fog.application.AppLoop;
import org.fog.application.AppModule;
import org.fog.application.Application;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.Sensor;
import org.fog.entities.Tuple;
import org.fog.utils.Config;
import org.fog.utils.FogEvents;
import org.fog.utils.FogUtils;
import org.fog.utils.NetworkUsageMonitor;
import org.fog.utils.TimeKeeper;
import org.fog.fanet.TaskScheduler;

public class Controller extends SimEntity {

	public static java.util.List<Tuple> taskWaitingPool = new java.util.ArrayList<>();

	public static boolean ONLY_CLOUD = false;

	private List<FogDevice> fogDevices;
	private List<Sensor> sensors;
	private List<Actuator> actuators;

	private Map<String, Application> applications;
	private Map<String, Integer> appLaunchDelays;
	private Map<String, ModulePlacement> appModulePlacementPolicy;

	protected Map<String, Integer> uavQueueTracker = new HashMap<>();
	private TaskScheduler taskScheduler;
	
	public long systemTotalScheduled = 0;
	public long systemTotalDropped = 0;
	public long totalSchedulingTimeNs = 0;
	public long schedulingInvocations = 0;

	// Metrics Tracking
	public double totalUAVUtilization = 0;
	public double totalCapacityUtilization = 0;
	public long totalUnassignedTasks = 0;
	public long totalGeneratedTasks = 0;

	public Controller(String name, List<FogDevice> fogDevices, List<Sensor> sensors, List<Actuator> actuators) {
		super(name);
		this.applications = new HashMap<String, Application>();
		setAppLaunchDelays(new HashMap<String, Integer>());
		setAppModulePlacementPolicy(new HashMap<String, ModulePlacement>());
		for (FogDevice fogDevice : fogDevices) {
			fogDevice.setControllerId(getId());
		}
		setFogDevices(fogDevices);
		setActuators(actuators);
		setSensors(sensors);
		connectWithLatencies();
	}

	private FogDevice getFogDeviceById(int id) {
		for (FogDevice fogDevice : getFogDevices()) {
			if (id == fogDevice.getId())
				return fogDevice;
		}
		return null;
	}

	private void connectWithLatencies() {
		for (FogDevice fogDevice : getFogDevices()) {
			FogDevice parent = getFogDeviceById(fogDevice.getParentId());
			if (parent == null)
				continue;
			double latency = fogDevice.getUplinkLatency();
			parent.getChildToLatencyMap().put(fogDevice.getId(), latency);
			parent.getChildrenIds().add(fogDevice.getId());
		}
	}

	@Override
	public void startEntity() {
		for (String appId : applications.keySet()) {
			if (getAppLaunchDelays().get(appId) == 0)
				processAppSubmit(applications.get(appId));
			else
				send(getId(), getAppLaunchDelays().get(appId), FogEvents.APP_SUBMIT, applications.get(appId));
		}
		send(getId(), Config.RESOURCE_MANAGE_INTERVAL, FogEvents.CONTROLLER_RESOURCE_MANAGE);
		send(getId(), Config.MAX_SIMULATION_TIME, FogEvents.STOP_SIMULATION);
		for (FogDevice dev : getFogDevices())
			sendNow(dev.getId(), FogEvents.RESOURCE_MGMT);
		
		// Start TF-PPO Sync loop every 10ms
		send(getId(), 10.0, FogEvents.TF_PPO_SYNC);
	}

	@Override
	public void processEvent(SimEvent ev) {
		org.cloudbus.cloudsim.core.CloudSimTags _tag_ = ev.getTag();
		if (_tag_ == FogEvents.APP_SUBMIT) {
			processAppSubmit(ev);

		} else if (_tag_ == FogEvents.TUPLE_FINISHED) {
			processTupleFinished(ev);

		} else if (_tag_ == FogEvents.TF_PPO_SYNC) {
			processTFPpoSync();

		} else if (_tag_ == FogEvents.CONTROLLER_RESOURCE_MANAGE) {
			manageResources();

		} else if (_tag_ == FogEvents.STOP_SIMULATION) {
			CloudSim.stopSimulation();
			// printTimeDetails();
			// printPowerDetails();
			// printCostDetails();
			// printNetworkUsageDetails();
			// printComputationLoadVariance();
			// System.exit(0); // Removing this so control returns to FANETSimulation.java

		} else if (_tag_ == FogEvents.TUPLE_ARRIVAL) {
			Tuple tuple = (Tuple) ev.getData();
			taskWaitingPool.add(tuple);
			if (taskWaitingPool.size() >= 100) {
				System.out.println("--- Batch of 100 tasks reached! Running Task Scheduling ---");
				runTaskScheduling();
			}

		} else {
			// ignore unexpected events
		}

	}
	
	private org.fog.fanet.TrajectoryModel trajectoryModel;

	public void setTrajectoryModel(org.fog.fanet.TrajectoryModel trajectoryModel) {
		this.trajectoryModel = trajectoryModel;
	}

	private void processTFPpoSync() {
		java.util.List<FogDevice> uavs = new java.util.ArrayList<>();
		java.util.List<FogDevice> mds = new java.util.ArrayList<>();
		for (FogDevice device : getFogDevices()) {
			if (device.getName().startsWith("uav")) {
				uavs.add(device);
			} else if (device.getName().startsWith("ground_device_")) {
				mds.add(device);
			}
		}
		
		if (this.trajectoryModel != null) {
			this.trajectoryModel.updateTrajectories(uavs, mds);
		}
		
		// Schedule next sync in 10ms
		send(getId(), 10.0, FogEvents.TF_PPO_SYNC);
	}

	private void printNetworkUsageDetails() {
		System.out
				.println("Total network usage = " + NetworkUsageMonitor.getNetworkUsage() / Config.MAX_SIMULATION_TIME);
	}

	private FogDevice getCloud() {
		for (FogDevice dev : getFogDevices())
			if (dev.getName().equals("cloud"))
				return dev;
		return null;
	}

	private void printCostDetails() {
		System.out.println("Cost of execution in cloud = " + getCloud().getTotalCost());
	}

	private void printPowerDetails() {
		for (FogDevice fogDevice : getFogDevices()) {
			System.out.println(fogDevice.getName() + " : Energy Consumed = " + fogDevice.getEnergyConsumption());
		}
	}

	private String getStringForLoopId(int loopId) {
		for (String appId : getApplications().keySet()) {
			Application app = getApplications().get(appId);
			for (AppLoop loop : app.getLoops()) {
				if (loop.getLoopId() == loopId)
					return loop.getModules().toString();
			}
		}
		return null;
	}

	private void printTimeDetails() {
		System.out.println("=========================================");
		System.out.println("============== RESULTS ==================");
		System.out.println("=========================================");
		System.out.println("EXECUTION TIME : "
				+ (Calendar.getInstance().getTimeInMillis() - TimeKeeper.getInstance().getSimulationStartTime()));
		System.out.println("=========================================");
		System.out.println("APPLICATION LOOP DELAYS");
		System.out.println("=========================================");
		for (Integer loopId : TimeKeeper.getInstance().getLoopIdToTupleIds().keySet()) {
			System.out.println(getStringForLoopId(loopId) + " ---> "
					+ TimeKeeper.getInstance().getLoopIdToCurrentAverage().get(loopId));
		}
		System.out.println("=========================================");
		System.out.println("TUPLE CPU EXECUTION DELAY");
		System.out.println("=========================================");
		for (String tupleType : TimeKeeper.getInstance().getTupleTypeToAverageCpuTime().keySet()) {
			System.out.println(
					tupleType + " ---> " + TimeKeeper.getInstance().getTupleTypeToAverageCpuTime().get(tupleType));
		}
		System.out.println("=========================================");
	}

	/**
	 * Metric 5 – Computation Load Variance
	 *
	 * Computes the continuous-time variance of CPU utilization across all UAV and
	 * terrestrial edge server nodes (i.e., every FogDevice that is not the cloud).
	 *
	 * Variance = E[U^2] - (E[U])^2 (population variance over all samples)
	 *
	 * Each sample represents the CPU utilization fraction [0,1] at one
	 * RESOURCE_MGMT_INTERVAL tick. All samples from all edge devices are pooled
	 * together so the result reflects the global load-balancing spread.
	 */
	private void printComputationLoadVariance() {
		System.out.println("=========================================");
		System.out.println("COMPUTATION LOAD VARIANCE (Metric 5)");
		System.out.println("=========================================");

		// Collect utilization samples from all edge devices (UAVs + terrestrial edge).
		// The cloud node is excluded because we only want to measure the edge layer.
		List<Double> allSamples = new java.util.ArrayList<>();
		for (FogDevice dev : getFogDevices()) {
			if (dev.getName().toLowerCase().equals("cloud"))
				continue;
			allSamples.addAll(dev.utilizationSamples);
		}

		if (allSamples.isEmpty()) {
			System.out.println("No utilization samples collected – variance cannot be computed.");
			System.out.println("=========================================");
			return;
		}

		// --- Two-pass variance: E[U] then E[(U - mean)^2] ---
		double sum = 0.0;
		for (double u : allSamples)
			sum += u;
		double mean = sum / allSamples.size();

		double sq = 0.0;
		for (double u : allSamples)
			sq += (u - mean) * (u - mean);
		double variance = sq / allSamples.size();
		double stddev = Math.sqrt(variance);

		// Per-device breakdown (mean & variance per device)
		for (FogDevice dev : getFogDevices()) {
			if (dev.getName().toLowerCase().equals("cloud"))
				continue;
			List<Double> s = dev.utilizationSamples;
			if (s.isEmpty()) {
				System.out.printf("  %-22s | samples: 0%n", dev.getName());
				continue;
			}
			double dSum = 0.0;
			for (double u : s)
				dSum += u;
			double dMean = dSum / s.size();
			double dSq = 0.0;
			for (double u : s)
				dSq += (u - dMean) * (u - dMean);
			double dVar = dSq / s.size();
			System.out.printf("  %-22s | samples: %4d | mean util: %.4f | variance: %.6f%n",
					dev.getName(), s.size(), dMean, dVar);
		}

		System.out.println("-----------------------------------------");
		System.out.printf("  Total samples       : %d%n", allSamples.size());
		System.out.printf("  Global mean util    : %.4f  (%.2f%%)%n", mean, mean * 100);
		System.out.printf("  Global variance     : %.6f%n", variance);
		System.out.printf("  Global std-dev      : %.6f%n", stddev);
		System.out.println("=========================================");
	}

	protected void manageResources() {
		send(getId(), Config.RESOURCE_MANAGE_INTERVAL, FogEvents.CONTROLLER_RESOURCE_MANAGE);
	}

	private void processTupleFinished(SimEvent ev) {
	}

	@Override
	public void shutdownEntity() {
	}

	public void submitApplication(Application application, int delay, ModulePlacement modulePlacement) {
		FogUtils.appIdToGeoCoverageMap.put(application.getAppId(), application.getGeoCoverage());
		getApplications().put(application.getAppId(), application);
		getAppLaunchDelays().put(application.getAppId(), delay);
		getAppModulePlacementPolicy().put(application.getAppId(), modulePlacement);
		for (Sensor sensor : sensors) {
			sensor.setApp(getApplications().get(sensor.getAppId()));
		}
		for (Actuator ac : actuators) {
			ac.setApp(getApplications().get(ac.getAppId()));
		}
		for (AppEdge edge : application.getEdges()) {
			if (edge.getEdgeType() == AppEdge.ACTUATOR) {
				String moduleName = edge.getSource();
				for (Actuator actuator : getActuators()) {
					if (actuator.getActuatorType().equalsIgnoreCase(edge.getDestination()))
						application.getModuleByName(moduleName).subscribeActuator(actuator.getId(),
								edge.getTupleType());
				}
			}
		}
	}

	public void submitApplication(Application application, ModulePlacement modulePlacement) {
		submitApplication(application, 0, modulePlacement);
	}

	private void processAppSubmit(SimEvent ev) {
		Application app = (Application) ev.getData();
		processAppSubmit(app);
	}

	private void processAppSubmit(Application application) {
		System.out.println(CloudSim.clock() + " Submitted application " + application.getAppId());
		FogUtils.appIdToGeoCoverageMap.put(application.getAppId(), application.getGeoCoverage());
		getApplications().put(application.getAppId(), application);
		ModulePlacement modulePlacement = getAppModulePlacementPolicy().get(application.getAppId());
		for (FogDevice fogDevice : fogDevices) {
			sendNow(fogDevice.getId(), FogEvents.ACTIVE_APP_UPDATE, application);
		}
		Map<Integer, List<AppModule>> deviceToModuleMap = modulePlacement.getDeviceToModuleMap();
		for (Integer deviceId : deviceToModuleMap.keySet()) {
			for (AppModule module : deviceToModuleMap.get(deviceId)) {
				sendNow(deviceId, FogEvents.APP_SUBMIT, application);
				sendNow(deviceId, FogEvents.LAUNCH_MODULE, module);
			}
		}
	}

	public void runTaskScheduling() {
		if (taskWaitingPool.isEmpty())
			return;

		// 1. Identify all FogDevices acting as UAVs and MDs
		java.util.List<FogDevice> uavs = new java.util.ArrayList<>();
		java.util.List<FogDevice> mds = new java.util.ArrayList<>();
		for (FogDevice device : getFogDevices()) {
			if (device.getName().startsWith("uav")) {
				uavs.add(device);
			} else if (device.getName().startsWith("ground_device")) {
				mds.add(device);
			}
		}

		// 3. Local Java Task Scheduling
		long startNs = System.nanoTime();
		if (taskScheduler != null) {
			taskScheduler.scheduleTasks(taskWaitingPool, uavs);
		} else {
			System.out.println("WARNING: No TaskScheduler configured in Controller!");
		}
		long endNs = System.nanoTime();
		this.totalSchedulingTimeNs += (endNs - startNs);
		this.schedulingInvocations++;

		// 4. Physically route the Tuples
		int scheduledInBatch = 0;
		int droppedInBatch = 0;
		for (Tuple task : taskWaitingPool) {
			if (task.assignedUavId != null) {
				sendNow(task.assignedUavId, FogEvents.TUPLE_ARRIVAL, task);
				scheduledInBatch++;
			} else {
				// System.out.println("Task " + task.getCloudletId() + " dropped (No UAVs in range or swamped)");
				droppedInBatch++;
			}
		}
		this.systemTotalScheduled += scheduledInBatch;
		this.systemTotalDropped += droppedInBatch;
		this.totalUnassignedTasks += droppedInBatch;
		this.totalGeneratedTasks += taskWaitingPool.size();

		// Calculate Utilizations for this batch
		double activeCount = 0;
		double capUtilSum = 0;
		if (!uavs.isEmpty()) {
			for (FogDevice uav : uavs) {
				if (uav.acceptedTasks.size() > 0) activeCount++;
				capUtilSum += (double) uav.acceptedTasks.size() / (double) uav.taskCapacity;
			}
			this.totalUAVUtilization += (activeCount / uavs.size());
			this.totalCapacityUtilization += (capUtilSum / uavs.size());
		}

		// Output time-series metrics for graphing system
		double currentRuntimeMs = (endNs - startNs) / 1000000.0;
		double currentUavUtil = uavs.isEmpty() ? 0 : (activeCount / uavs.size());
		double currentCapUtil = uavs.isEmpty() ? 0 : (capUtilSum / uavs.size());
		
		long rqLen = 0, inv = 0, re = 0;
		if (taskScheduler instanceof org.fog.fanet.DynamicRepairScheduler) {
			org.fog.fanet.DynamicRepairScheduler drs = (org.fog.fanet.DynamicRepairScheduler) taskScheduler;
			rqLen = drs.getTotalRepairQueueLength(); // Cumulative, but can be plotted
			inv = drs.getTotalInvalidAssignments();
			re = drs.getTotalReassignedTasks();
		}

		System.out.println(String.format("[METRIC_EPOCH] Epoch:%d, Generated:%d, Scheduled:%d, Dropped:%d, RuntimeMs:%.4f, UavUtil:%.4f, CapUtil:%.4f, RepairQ:%d, Invalid:%d, Reassigned:%d",
				schedulingInvocations, taskWaitingPool.size(), scheduledInBatch, droppedInBatch, currentRuntimeMs, currentUavUtil, currentCapUtil, rqLen, inv, re));

		// 5. Clear the pool for the next batch
		taskWaitingPool.clear();
	}

	private int getCloudId() {
		for (FogDevice device : getFogDevices()) {
			if (device.getName().toLowerCase().contains("cloud")) {
				return device.getId();
			}
		}
		return -1;
	}

	public List<FogDevice> getFogDevices() {
		return fogDevices;
	}

	public void setFogDevices(List<FogDevice> fogDevices) {
		this.fogDevices = fogDevices;
	}

	public Map<String, Integer> getAppLaunchDelays() {
		return appLaunchDelays;
	}

	public void setAppLaunchDelays(Map<String, Integer> appLaunchDelays) {
		this.appLaunchDelays = appLaunchDelays;
	}

	public Map<String, Application> getApplications() {
		return applications;
	}

	public void setApplications(Map<String, Application> applications) {
		this.applications = applications;
	}

	public List<Sensor> getSensors() {
		return sensors;
	}

	public void setSensors(List<Sensor> sensors) {
		for (Sensor sensor : sensors)
			sensor.setControllerId(getId());
		this.sensors = sensors;
	}

	public List<Actuator> getActuators() {
		return actuators;
	}

	public void setActuators(List<Actuator> actuators) {
		this.actuators = actuators;
	}

	public Map<String, ModulePlacement> getAppModulePlacementPolicy() {
		return appModulePlacementPolicy;
	}

	public void setAppModulePlacementPolicy(Map<String, ModulePlacement> appModulePlacementPolicy) {
		this.appModulePlacementPolicy = appModulePlacementPolicy;
	}

	public void setTaskScheduler(TaskScheduler taskScheduler) {
		this.taskScheduler = taskScheduler;
	}
}