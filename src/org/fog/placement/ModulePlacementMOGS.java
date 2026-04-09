package org.fog.placement;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.cloudbus.cloudsim.core.CloudSim;
import org.fog.application.AppModule;
import org.fog.application.Application;
import org.fog.entities.Actuator;
import org.fog.entities.FogDevice;
import org.fog.entities.Sensor;

public class ModulePlacementMOGS extends ModulePlacement {

    protected ModuleMapping moduleMapping;

    public ModulePlacementMOGS(List<FogDevice> fogDevices, List<Sensor> sensors, List<Actuator> actuators, 
                               Application application, ModuleMapping moduleMapping) {
        this.setFogDevices(fogDevices);
        this.setApplication(application);
        this.setModuleMapping(moduleMapping);
        this.setModuleToDeviceMap(new HashMap<String, List<Integer>>());
        this.setDeviceToModuleMap(new HashMap<Integer, List<AppModule>>()); 
        this.setModuleInstanceCountMap(new HashMap<Integer, Map<String, Integer>>());
        
        mapModules();
    }

    public ModuleMapping getModuleMapping() {
        return moduleMapping;
    }

    public void setModuleMapping(ModuleMapping moduleMapping) {
        this.moduleMapping = moduleMapping;
    }

    @Override
    public void mapModules() { 
        // 1. Map predefined modules (like storage_module -> cloud, sensor -> ground)
        for(String deviceName : getModuleMapping().getModuleMapping().keySet()){
            for(String moduleName : getModuleMapping().getModuleMapping().get(deviceName)){
                int deviceId = CloudSim.getEntityId(deviceName);
                placeModule(moduleName, deviceId);
            }
        }

        // 2. Pre-load the 'processing_module' on ALL UAVs. 
        // The Controller will handle the actual MOGS routing of tasks to these UAVs at runtime.
        for (FogDevice device : getFogDevices()) {
            if (device.getName().startsWith("uav_")) {
                placeModule("processing_module", device.getId());
            }
        }
    }
    
    // Standard iFogSim module placement logic
    private void placeModule(String moduleName, int deviceId) {
        if(!getDeviceToModuleMap().containsKey(deviceId)){
            getDeviceToModuleMap().put(deviceId, new ArrayList<AppModule>());
        }
        
        AppModule appModule = getApplication().getModuleByName(moduleName);
        if(appModule == null) return;
        
        getDeviceToModuleMap().get(deviceId).add(appModule);
        
        if(!getModuleToDeviceMap().containsKey(moduleName)){
            getModuleToDeviceMap().put(moduleName, new ArrayList<Integer>());
        }
        getModuleToDeviceMap().get(moduleName).add(deviceId);
    }
}