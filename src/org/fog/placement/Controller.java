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
	}

	@Override
	public void processEvent(SimEvent ev) {
		org.cloudbus.cloudsim.core.CloudSimTags _tag_ = ev.getTag();
		if (_tag_ == FogEvents.APP_SUBMIT) {
			processAppSubmit(ev);

		} else if (_tag_ == FogEvents.TUPLE_FINISHED) {
			processTupleFinished(ev);

		} else if (_tag_ == FogEvents.CONTROLLER_RESOURCE_MANAGE) {
			manageResources();

		} else if (_tag_ == FogEvents.STOP_SIMULATION) {
			CloudSim.stopSimulation();
			printTimeDetails();
			printPowerDetails();
			printCostDetails();
			printNetworkUsageDetails();
			System.exit(0);

		} else if (_tag_ == FogEvents.TUPLE_ARRIVAL) {
			Tuple tuple = (Tuple) ev.getData();
			taskWaitingPool.add(tuple);
			if (taskWaitingPool.size() >= 20) {
				System.out.println("--- Batch of 20 tasks reached! Running MOGS Matching ---");
				runFanetMOGSScheduling();
			}

		} else {
			// ignore unexpected events
		}

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

	public void runFanetMOGSScheduling() {
		if (taskWaitingPool.isEmpty())
			return;

		// 1. Identify all FogDevices acting as UAVs
		java.util.List<FogDevice> uavs = new java.util.ArrayList<>();
		for (FogDevice device : getFogDevices()) {
			if (device.getName().startsWith("uav")) {
				uavs.add(device);
			}
		}

		// 2. Build Preferences via PPO + utility math
		org.fog.fanet.PreferenceBuilder.buildPreferences(taskWaitingPool, uavs);

		// 3. Run MOGS Matching
		org.fog.fanet.MOGSMatching.runMatching(taskWaitingPool, uavs);

		// 4. Update persistent queue tracker with real assigned task counts
		for (FogDevice uav : uavs) {
			this.uavQueueTracker.put(uav.getName(), uav.acceptedTasks.size());
		}

		// 5. Physically route the Tuples
		for (Tuple task : taskWaitingPool) {
			if (task.assignedUavId != null) {
				sendNow(task.assignedUavId, FogEvents.TUPLE_ARRIVAL, task);
			} else {
				sendNow(getCloudId(), FogEvents.TUPLE_ARRIVAL, task);
			}
		}

		// 6. Clear the pool for the next batch
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
}