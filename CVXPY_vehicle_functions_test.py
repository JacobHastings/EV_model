import cvxpy as cp
import numpy as np
import random
import matplotlib.pyplot as plt
from Vehicle_class import Vehicle
from Charger_class import Charger
from Manager_class import Manager
import json

interval = 3600
sim_end = 86400*1
T = int(sim_end/interval)
flag_maintain_SOC = True
maintain_constraints_idx = []

M = Manager()

##############################################################################
######################### Initialize Vehicles ################################
##############################################################################
Vehicles = []
vehicle_count = 100
start_SOC = 50 * np.ones(vehicle_count)

for i in range(vehicle_count):
    V = Vehicle()
    
    V.work_start = V.work_start + (0.5 * random.randint(-2,8))
    V.work_duration = V.work_duration - (0.5 * random.randint(-1,3))
    V.commute_distance = V.commute_distance * (1 + (0.1*random.randint(-1,2)))
    
    # Fast charge test
    # # Default C-rate approx 0.05
    # V.maximum_charge_rate = V.maximum_charge_rate * 40
    # # Default start = 5
    # V.work_start = V.work_start - 3
    # # Default duration is 11.5
    # V.work_duration = V.work_duration + 5
    
    # Short work duration test
    # V.commute_duration += (((V.work_duration*3600) *0.98) /2)
    # V.work_start += V.work_duration*0.98/2
    # V.work_duration = V.work_duration * 0.02
    
    V.index = i
    V.set_day_schedule(sim_end)
    V.battery_SOC = start_SOC[i] * (1 + (0.1 * random.randint(-2,2)))
    V.update_capacity()
    V.update_log()
    Vehicles.append(V)

M.vehicles = Vehicles
M.initialize_chargers_from_vehicles()
M.last_setting_change = np.zeros(len(M.chargers))

######################### Update LMP Estimates ################################
LMP_test = 20 * np.ones(T)
for i in range(T):
    hour = int(i*interval/3600)
    if 0 <= (hour%24) < 6:
        LMP_test[i] = LMP_test[i] * 0.5
    if 16 <= (hour%24) < 20:
        LMP_test[i] = LMP_test[i] * 2
M.LMP_est = LMP_test

##################### Optimize Schedule & Form Bids ###########################
M.initial_optimization()

planned_schedule_fleet = M.charge_schedule_fleet.copy()

################# Provide Cleared Quantities for Fleet ########################
# Alter schedule for testing
altered_schedule = np.zeros((T,vehicle_count))
altered_schedule_fleet = np.zeros(T)
for i in range(T):
    altered_schedule_fleet[i] = M.charge_schedule_fleet[i] + (1000*random.uniform(-1*sum(M.low_energy_used[i,:]),sum(M.high_energy_used[i,:])))
M.charge_schedule_fleet = altered_schedule_fleet

####################### Optimize Final DA Schedule ############################
M.final_optimization()

########################### Simulate Charging #################################
managed_time = 0
sim_interval = 300
while managed_time < 86400:
    M.simulate_scheduled_DAM(sim_interval)
    managed_time += sim_interval
###############################################################################

plot_time = np.ones(T)
for i in range(T):
    plot_time[i] = i*interval/3600

plot_m_time = []
plot_m_load = []
for i in range(len(M.load_log)):
    if i>0:
        plot_m_time.append((M.load_log[i][0]/3600))
        plot_m_load.append(M.load_log[i][1])

plt.plot(plot_time,planned_schedule_fleet)
plt.plot(plot_time,planned_schedule_fleet+(1000*np.sum(M.high_energy_used,axis=1)))
plt.plot(plot_time,planned_schedule_fleet-(1000*np.sum(M.low_energy_used,axis=1)))
plt.plot(plot_time,altered_schedule_fleet)
plt.plot(plot_m_time,plot_m_load)
plt.xlabel("Time (hour)")
plt.ylabel("Fleet Schedule (Watts)")
legend = ['Planned','High Margin','Low Margin','Final Cleared','Managed Charging']
plt.legend(legend)
plt.grid()
plt.show()

# Subsequent Day test - passed
# M.initial_optimization()

# planned_schedule_fleet2 = M.charge_schedule_fleet.copy()

# for i in range(T):
#     altered_schedule_fleet[i] = M.charge_schedule_fleet[i] + (1000*random.uniform(-1*sum(M.low_energy_used[i,:]),sum(M.high_energy_used[i,:])))
# M.charge_schedule_fleet = altered_schedule_fleet

# M.final_optimization()

# while managed_time < (86400*2):
#     M.simulate_scheduled_DAM(300)
#     managed_time += 300