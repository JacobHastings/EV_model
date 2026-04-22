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

Vehicles = []
vehicle_count = 100
start_SOC = 50 * np.ones(vehicle_count)

for i in range(vehicle_count):
    V = Vehicle()
    
    V.work_start = V.work_start + (0.5 * random.randint(-2,8))
    V.work_duration = V.work_duration - (0.5 * random.randint(-1,3))
    V.commute_distance = V.commute_distance * (1 + (0.1*random.randint(-1,2)))
    
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

LMP_test = 20 * np.ones(T)
for i in range(T):
    hour = int(i*interval/3600)
    if 0 <= (hour%24) < 6:
        LMP_test[i] = LMP_test[i] * 0.5
    if 16 <= (hour%24) < 20:
        LMP_test[i] = LMP_test[i] * 2
LMP = cp.Parameter(T)
LMP.value = LMP_test

# Only works for schedules set by Vehicle.set_day_schedule(), not for otherwise assigned or edited schedules
for V in Vehicles:
    if (2*V.commute_duration)+(V.work_duration*3600) < (2*interval):
        print("********************* Warning: Vehicle schedule violates assumptions; decrease interval *********************")

##############################################################################
#                      Begin Schedule Optimization                           #
##############################################################################

constraints = []

hours_per_interval = cp.Parameter(value=interval/3600)

A = cp.Parameter((T,T),nonneg=True)
A.value = np.ones((T,T))

SOC_max = cp.Parameter(nonneg=True)
SOC_max.value = 100
SOC_min = cp.Parameter(nonneg=True)
SOC_min.value = 0
SOC_start = cp.Parameter(vehicle_count,nonneg=True)
for i in range(vehicle_count):
    start_SOC[i] = Vehicles[i].SOC_log[0][1]
SOC_start.value = start_SOC

Charge_schedule = cp.Variable((T,vehicle_count))
constraints += [Charge_schedule >= 0, Charge_schedule <= V.maximum_charge_rate]

charge_available = cp.Parameter((T,vehicle_count),nonneg=True)
charge_available.value = np.zeros((T,vehicle_count))

driving_loss = cp.Parameter((T,vehicle_count),nonpos=True)
driving_loss.value = np.zeros((T,vehicle_count))

SOC = cp.Variable((T,vehicle_count),nonneg=True)

for V in Vehicles:
    for i in range(T):
        time = i * interval
        end = time + interval - 1
        # Reset then determine location at time i
        location = 'NONE'
        next_location = 'NONE'
        next_time = sim_end
        for entry in range(len(V.schedule)-1):
            if V.schedule[entry][0] <= time:
                location = V.schedule[entry][1]
                next_time = V.schedule[entry+1][0]
                next_location = V.schedule[entry+1][1]
            # Check for 2 location changes in this interval, record last one if 'HOME'
            elif V.schedule[entry+1][0] <= end and V.schedule[entry+1][1]=='HOME':
                next_time = V.schedule[entry+1][0]
                next_location = V.schedule[entry+1][1]
            else:
                break
        
        # Charge_available gives the decimal fraction of time plugged in
        if location == 'HOME':
            if next_time >= end:
                charge_available.value[i,V.index] = 1
            else:
                charge_available.value[i,V.index] = (next_time - time) / interval
        
        if next_location == 'HOME' and next_time < end:
            charge_available.value[i,V.index] = (end - next_time) / interval
        
        # Determine driving SOC losses
        for entry in range(len(V.schedule)-1):
            if (V.schedule[entry][1] == 'DRIVING_WORK') and (time<=V.schedule[entry][0]<=end) and i<(T-1):
                if V.schedule[entry][0] == time:
                    driving_loss.value[i,V.index] = -100*((V.commute_distance/V.mileage_efficiency)/V.battery_size)
                else:
                    driving_loss.value[i+1,V.index] = -100*((V.commute_distance/V.mileage_efficiency)/V.battery_size)
                    
            # Backup logic if time outside of home becomes too small, edge cases untested
            # if (V.schedule[entry][1] == 'DRIVING_HOME') and (V.schedule[entry+1][1] == 'HOME') and ((time+interval) <= V.schedule[entry+1][0] <= (end+interval)):
            #     if V.schedule[entry+1][0] == (end+interval) and i<T-1:
            #         driving_loss.value[i+1,V.index] = -100*((V.commute_distance/V.mileage_efficiency)/V.battery_size)
            #     else:
            #         driving_loss.value[i,V.index] = -100*((V.commute_distance/V.mileage_efficiency)/V.battery_size)
                        
            if (V.schedule[entry][1] == 'DRIVING_HOME') and (time<=V.schedule[entry][0]<=end):
                if i!=0:
                    if V.schedule[entry+1][0] > end:
                        driving_loss.value[i,V.index] = -100*((V.commute_distance/V.mileage_efficiency)/V.battery_size)
                    else:
                        driving_loss.value[i-1,V.index] = -100*((V.commute_distance/V.mileage_efficiency)/V.battery_size)
                # Otherwise the loss has already been applied
        
        # Construct lower triangular matrix for running SOC calculation
        for j in range(T):
            if j>i:
                A.value[i,j] = 0

    # Convert charge_available to watts
    constraints += [Charge_schedule[:,V.index] <= charge_available[:,V.index]*V.maximum_charge_rate]
    
    SOC_per_watt_hr = 100 / (V.battery_size*1000)
    
    # ((A @ ((Charge_schedule*(interval/3600)*V.charging_efficiency*SOC_per_watt_hr)+driving_loss))+SOC_start) gives vector of SOC
    constraints += [SOC[:,V.index] == ((A @ ((Charge_schedule[:,V.index]*hours_per_interval*V.charging_efficiency*SOC_per_watt_hr)+driving_loss[:,V.index]))+SOC_start[V.index])]
    constraints += [SOC[:,V.index] <= SOC_max, SOC[:,V.index] >= SOC_min]
    
    if flag_maintain_SOC:
        constraints += [SOC[T-1,V.index] >= SOC_start[V.index]]
        maintain_constraints_idx.append(len(constraints)-1) 
        

A1 = cp.Parameter(nonneg=True)      # Price
A1.value = 1    #1
A2 = cp.Parameter(nonneg=True)      # SOC
A2.value = 50 #50
A3 = cp.Parameter(nonneg=True)      # Smoothing avg
A3.value = 100 #100
A4 = cp.Parameter(nonneg=True)      # Smoothing diff
A4.value = 200 #200
A5 = cp.Parameter(nonneg=True)      # 20-80 rule
A5.value = 0

Charge_schedule_fleet = cp.Variable(T)
constraints += [Charge_schedule_fleet == cp.sum(Charge_schedule,axis=1)]


objective = cp.Minimize(
    A1 * cp.sum(cp.multiply(LMP,Charge_schedule_fleet))
    +
    A2 * cp.sum(100 - SOC)
    +
    A3 * cp.norm(Charge_schedule_fleet)
    +
    A4 * cp.norm(Charge_schedule_fleet[0:T-1] - Charge_schedule_fleet[1:T])
    + 
    A5 
    )



prob = cp.Problem(objective,constraints)

prob.solve(solver=cp.GUROBI)

planned_schedule = Charge_schedule.value
planned_schedule_fleet = Charge_schedule_fleet.value

print("status:", prob.status)
print("optimal value", prob.value)
# print("Charging Schedule (Watts)")
# print(Charge_schedule.value)
# print("SOC schedule (%)")
# print(SOC.value)

# Used to check optimization values to tune A_constants
O1 = A1.value * sum(LMP.value * Charge_schedule_fleet.value)
O2 = A2.value * sum(sum((100*np.ones((T,vehicle_count))) - SOC.value))
O3 = A3.value * np.linalg.norm(Charge_schedule_fleet.value)
O4 = A4.value * np.linalg.norm(Charge_schedule_fleet[0:T-1].value - Charge_schedule_fleet[1:T].value)
O5 = A5.value 

print("Price term:\t\t\t\t", O1)
print("SOC term:\t\t\t\t", O2)
print("Smoothing avg term:\t\t", O3)
print("Smoothing diff term:\t", O4)
# print("20-80 rule term:\t\t", O5)

##############################################################################
plot_time = np.ones(T)
for i in range(T):
    plot_time[i] = i*interval/3600
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel("Time (hour)")
ax1.set_ylabel("Charging Schedule (W)", color=color)
ax1.plot(plot_time,Charge_schedule.value, color=color)
plt.ylim(-50,1800)
plt.grid()
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx() 

color = 'tab:gray'
ax2.set_ylabel("SOC Schedule (%)", color=color)
ax2.plot(plot_time,SOC.value, color=color)
plt.ylim(0,100)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Optimized Schedule")
plt.show()
##############################################################################

plt.plot(plot_time,SOC.value)
plt.xlabel("Time (hour)")
plt.ylabel("SOC Schedule (%)")
# plt.ylim(0,100)
plt.grid()
plt.show()

plt.plot(plot_time,Charge_schedule.value)
plt.xlabel("Time (hour)")
plt.ylabel("Charging Schedule (W)")
plt.ylim(-50,1800)
plt.grid()
plt.show()

plt.plot(plot_time,Charge_schedule_fleet.value)
plt.xlabel("Time (hour)")
plt.ylabel("Fleet Charging Schedule (W)")
# plt.ylim(-50,1800)
plt.grid()
plt.show()


##############################################################################
#                       Begin Bidding Formulation                            #
##############################################################################
extra_energy_margin = 1.0      # >1 for allowable issues; =1 for exact margins
P_buffer = 2
bid_slope = 0.0000035 * vehicle_count

bids = []

power_available = np.zeros((T,vehicle_count))
power_used = np.zeros((T,vehicle_count))
high_energy_margin = np.zeros(vehicle_count)
low_energy_margin = np.zeros(vehicle_count)

for V in Vehicles:
    # Determine total kWh margins for 24-hour period
    highest_scheduled_SOC = max(SOC.value[:,V.index])
    high_energy_margin[V.index] = extra_energy_margin * (((100 - highest_scheduled_SOC)/100) * V.battery_size) / V.charging_efficiency # kWh that can be increased
    lowest_scheduled_SOC = min(SOC.value[:,V.index])
    low_energy_margin[V.index] = extra_energy_margin * ((lowest_scheduled_SOC/100) * V.battery_size) / V.charging_efficiency           # kWh that can be reduced
    
    for i in range(T):
        power_available[i,V.index] = (charge_available[i,V.index].value * V.maximum_charge_rate) - Charge_schedule[i,V.index].value
        power_used[i,V.index] = Charge_schedule[i,V.index].value
    
fleet_power_available = sum(power_available.transpose())
fleet_power_used = sum(power_used.transpose())
fleet_low_energy_margin = sum(low_energy_margin)
fleet_high_energy_margin = sum(high_energy_margin)

hours_per_interval = cp.Parameter(value=interval/3600)
LMP_avg = cp.Parameter(value = sum(LMP.value) / T)

B1 = cp.Parameter(nonneg=True)
B2 = cp.Parameter(nonneg=True)
B3 = cp.Parameter(nonneg=True)
B4 = cp.Parameter(nonneg=True)

# Fully utilize high margins
B1.value = 1000     #1000
# Fully utilize low margins
B2.value = 1000     #1000
# Spread flexibility across time steps
B3.value = 1        #1
# Center flexibility around high/low price times
B4.value = 0.02     #0.02

bid_constraints = []

high_energy_used = cp.Variable((T,vehicle_count), nonneg=True)
low_energy_used = cp.Variable((T,vehicle_count), nonneg=True)

power_available_P = cp.Parameter((T,vehicle_count))
power_available_P.value = power_available
power_used_P = cp.Parameter((T,vehicle_count))
power_used_P.value = power_used
high_energy_margin_P = cp.Parameter(vehicle_count)
high_energy_margin_P.value = high_energy_margin
low_energy_margin_P = cp.Parameter(vehicle_count)
low_energy_margin_P.value = low_energy_margin

# Hard limit, not exceeding available margins per vehicle
bid_constraints += [cp.sum(high_energy_used,axis=0) * hours_per_interval <= high_energy_margin_P]
bid_constraints += [cp.sum(low_energy_used,axis=0) * hours_per_interval <= low_energy_margin_P]

# Can't increase usage by more than available per vehicle
bid_constraints += [high_energy_used <= ((power_available_P / 1000) * hours_per_interval)]
# Can't decrease usage by more than what was used per vehicle
bid_constraints += [low_energy_used <= ((power_used_P / 1000) * hours_per_interval)]


bid_objective = cp.Minimize(
    B1 * ( cp.sum(high_energy_margin_P - cp.sum(high_energy_used,axis=0)))
    +
    B2 * cp.sum(low_energy_margin_P - cp.sum(low_energy_used,axis=0))
    +
    B3 * (cp.sum_squares(low_energy_used) + cp.sum_squares(high_energy_used))
    +
    B4 * -1* cp.sum(cp.multiply((cp.sum((low_energy_used+high_energy_used),axis=1)), cp.abs(LMP-LMP_avg)))
    )


bid_prob = cp.Problem(bid_objective,bid_constraints)

bid_prob.solve(solver=cp.GUROBI)
# bid_prob.solve()


print("status:", bid_prob.status)
print("optimal value", bid_prob.value)

# Watts; matches fleet_power_used
fleet_low_power_used = 1000 * hours_per_interval.value * np.sum(low_energy_used.value,axis=1)
fleet_high_power_used = 1000 * hours_per_interval.value * np.sum(high_energy_used.value,axis=1)


for i in range(T):
    bid = []
    Q_plan = fleet_power_used[i]
    P_plan = LMP[i].value
    if fleet_low_power_used[i] > 0:
        bid.append([np.round(Q_plan - fleet_low_power_used[i]), np.round(P_plan+P_buffer+(fleet_low_power_used[i]*bid_slope))])
    bid.append([np.round(Q_plan), P_plan+P_buffer])
    bid.append([np.round(Q_plan), P_plan-P_buffer])
    if fleet_high_power_used[i] > 0:
        bid.append([np.round(Q_plan+fleet_high_power_used[i]), np.round(P_plan-P_buffer-(fleet_high_power_used[i]*bid_slope))])
    bids.append(bid)

max_bid_len = 0
for bid in bids:
    max_bid_len = max(max_bid_len,len(bid))

cosim_bus_mul = 4021.74
bids_out = {}
bids_out['Q_bid'] = np.zeros((T,max_bid_len))
bids_out['P_bid'] = np.zeros((T,max_bid_len))
bids_out['constant_MVAR'] = np.zeros(T)
bids_out['constant_MW'] = np.zeros(T)
for i in range(T):
    bid_count = 0
    if len(bids[i]) > 2:
        for j in range(len(bids[i])):
            bids_out['Q_bid'][i,j] = (bids[i][j][0] / 1000000) * cosim_bus_mul
            bids_out['P_bid'][i,j] = bids[i][j][1]
            
bids_out['Q_bid'] = bids_out['Q_bid'].tolist()
bids_out['P_bid'] = bids_out['P_bid'].tolist()
bids_out['constant_MVAR'] = bids_out['constant_MVAR'].tolist()
bids_out['constant_MW'] = bids_out['constant_MW'].tolist()

M.bids_DAM = bids_out

# with open('bid_dump.json','w') as json_file_out:
#     json.dump(bids_out,json_file_out, indent=4)

##############################################################################
#               Match Cleared Quantities to Final Schedule                   #
##############################################################################
Initial_charge_schedule = cp.Parameter((T,vehicle_count), value = Charge_schedule.value)

# Just matches plan for now, alter later
Cleared_charge_schedule_fleet = cp.Parameter(T)

# Alter schedule for testing
altered_schedule = np.zeros((T,vehicle_count))
altered_schedule_fleet = np.zeros(T)
for i in range(T):
    altered_schedule_fleet[i] = Charge_schedule_fleet[i].value + (1000*random.uniform(-1*sum(low_energy_used[i,:].value),sum(high_energy_used[i,:].value)))
    
Cleared_charge_schedule_fleet.value = altered_schedule_fleet

cleared_constraints = constraints.copy()
# Remove requirement to end at same SOC as start
for index in range(len(maintain_constraints_idx)-1,-1,-1):
    del cleared_constraints[maintain_constraints_idx[index]]
# Add cleared quantity schedule constraints
cleared_constraints += [Cleared_charge_schedule_fleet == cp.sum(Charge_schedule,axis=1)]

C1 = cp.Parameter(nonneg=True)
C2 = cp.Parameter(nonneg=True)
C3 = cp.Parameter(nonneg=True)
C4 = cp.Parameter(nonneg=True)

# Minimize deviations from original plan
C1.value = 1
# Placeholder objective term
C2.value = 1
# 
C3.value = 1
# 
C4.value = 1

cleared_objective = cp.Minimize(
    C1 * cp.sum_squares(Charge_schedule - Initial_charge_schedule)
    +
    C2 
    +
    C3
    +
    C4
    )

cleared_prob = cp.Problem(cleared_objective,cleared_constraints)

cleared_prob.solve(solver=cp.GUROBI)

print("status:", cleared_prob.status)
print("optimal value", cleared_prob.value)

final_schedule = Charge_schedule.value

M.charge_schedule = final_schedule

print("Maximum deviation from cleared values:",(max(abs(Charge_schedule_fleet.value-Cleared_charge_schedule_fleet.value))),"Watts")

plt.plot(plot_time,Charge_schedule_fleet.value-planned_schedule_fleet)
plt.xlabel("Time (hour)")
plt.ylabel("Market Cleared Fleet Deviation (Watts)")
plt.grid()
plt.show()

plt.plot(plot_time,planned_schedule-final_schedule)
plt.xlabel("Time (hour)")
plt.ylabel("Vehicle Deviation from Plan (Watts)")
plt.grid()
plt.show()

plt.plot(plot_time,Charge_schedule.value)
plt.xlabel("Time (hour)")
plt.ylabel("Final Vehicle Schedules (Watts)")
plt.grid()
plt.show()

# plt.plot(plot_time,planned_schedule_fleet)
# plt.plot(plot_time,planned_schedule_fleet+(1000*np.sum(high_energy_used.value,axis=1)))
# plt.plot(plot_time,planned_schedule_fleet-(1000*np.sum(low_energy_used.value,axis=1)))
# plt.plot(plot_time,Charge_schedule_fleet.value)
# plt.xlabel("Time (hour)")
# plt.ylabel("Fleet Schedule (Watts)")
# legend = ['Planned','High Margin','Low Margin','Final Cleared']
# plt.legend(legend)
# plt.grid()
# plt.show()

managed_time = 0
while managed_time < 86400:
    M.simulate_scheduled_DAM(300)
    managed_time += 300

plot_m_time = []
plot_m_load = []
for i in range(len(M.load_log)):
    plot_m_time.append(M.load_log[i][0]/3600)
    plot_m_load.append(M.load_log[i][1])

# plt.plot(plot_m_time,plot_m_load)
# plt.xlabel("Time (hour)")
# plt.ylabel("Managed Charging load (Watts)")
# plt.grid()
# plt.show()

plt.plot(plot_time,planned_schedule_fleet)
plt.plot(plot_time,planned_schedule_fleet+(1000*np.sum(high_energy_used.value,axis=1)))
plt.plot(plot_time,planned_schedule_fleet-(1000*np.sum(low_energy_used.value,axis=1)))
plt.plot(plot_time,Charge_schedule_fleet.value)
plt.plot(plot_m_time,plot_m_load)
plt.xlabel("Time (hour)")
plt.ylabel("Fleet Schedule (Watts)")
legend = ['Planned','High Margin','Low Margin','Final Cleared','Managed Charging']
plt.legend(legend)
plt.grid()
plt.show()

