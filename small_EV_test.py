import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd
import numpy as np
import json
import glmanip

from Vehicle_class import Vehicle
from Charger_class import Charger


def clean_timestamp(dataframe):
    dataframe['# timestamp'] = [datetime.datetime.strptime(date.split(' PDT')[0], "%Y-%m-%d %H:%M:%S") for date in dataframe['# timestamp']]
    
def clean_timestamp_EST(dataframe):
    dataframe['# timestamp'] = [datetime.datetime.strptime(date.split(' EST')[0], "%Y-%m-%d %H:%M:%S") for date in dataframe['# timestamp']]


def output_from_gridlabd():
    filename = "C:/Users/jacob/Documents/MatpowerWrapper/EVtest/driverfile_charger.csv"
    raw_data_charger1 = pd.read_csv(filename, skiprows=8, sep=',')
    clean_timestamp(raw_data_charger1)
    filename = "C:/Users/jacob/Documents/MatpowerWrapper/EVtest/driverfile_charger2.csv"
    raw_data_charger2 = pd.read_csv(filename, skiprows=8, sep=',')
    clean_timestamp(raw_data_charger2)
    filename = "C:/Users/jacob/Documents/MatpowerWrapper/EVtest/driverfile_house.csv"
    raw_data_house = pd.read_csv(filename, skiprows=8, sep=',')
    clean_timestamp(raw_data_house)
    
    time = raw_data_charger1['# timestamp']
    house_load = raw_data_house['total_load']
    EV1_SOC = raw_data_charger1[' battery_SOC']
    EV1_charging = raw_data_charger1[' actual_charge_rate']
    EV2_charging = raw_data_charger2[' actual_charge_rate']
    combined_charging = EV1_charging + EV2_charging
    
    # Convert timestamps into seconds since start of simulation
    # plot_time = (np.array(time)).astype(int) / 10**9    #default is nanoseconds
    # plot_time = plot_time - plot_time[0]
    plot_time = (np.array(time)).astype(float) / 10**9   #default is nanoseconds
    plot_time = plot_time - plot_time[0]
    plot_time = plot_time.astype(int)
    
    plot_charge = np.array(combined_charging)
    plot_house = np.array(house_load) * 1000
    
    return plot_time, plot_charge, plot_house

def output_from_gridlabd_v2():
    filename = "C:/Users/jacob/Documents/MatpowerWrapper/tesp/examples/capabilities/feeder-generator/EV_charger_rate_output.csv"
    raw_data_charger = pd.read_csv(filename,skiprows=8)
    clean_timestamp_EST(raw_data_charger)
    
    time = raw_data_charger['# timestamp']
    # Convert timestamps into seconds since start of simulation
    plot_time = (np.array(time)).astype(float) / 10**9   #default is nanoseconds
    plot_time = plot_time - plot_time[0]
    plot_time = plot_time.astype(int)
    
    data_charger = np.array(raw_data_charger)
    data_charger = data_charger[:,1:]
    
    plot_charge = np.zeros(len(plot_time))
    for i in range(data_charger.shape[0]):
        temp_sum = 0
        for j in range(data_charger.shape[1]):
            temp_sum += data_charger[i,j]
        plot_charge[i] = temp_sum
    
    return plot_time, plot_charge

def agregate_loads(Chargers,sim_end,interval):
    plot_time = list(range(0,sim_end,interval))
    plot_load = [0.0] * len(plot_time)
    for j in range(len(plot_time)):
        for C in Chargers[0:]:
            for i in range(len(C.load_log)):
                if C.load_log[i][0] >= plot_time[j]:
                    plot_load[j] += C.load_log[i][1]
                    break
    return plot_time, plot_load

def numerical_integration(X,Y):
    result=float(0.0)
    for i in range(len(X)-1):
        result += ((Y[i]+Y[i+1])/2)*((X[i+1]-X[i]))
    return result

def difference_loads(time1,time2,load1,load2,sim_end,interval):
    plot_time = list(range(0,sim_end,interval))
    plot_load = [0.0] * len(plot_time)
    for j in range(len(plot_time)):
        for i in range(len(time1)):
            if time1[i] >= plot_time[j]:
                plot_load[j]+=load1[i]
                break
        for i in range(len(time2)):
            if time2[i] >= plot_time[j]:
                plot_load[j]-=load2[i]
                break
    
    return plot_time, plot_load
        
##############################################################################
#################################### Main ####################################
##############################################################################

plot_time_d, plot_charge_d, plot_house_d = output_from_gridlabd()

time = [0]
load = [0.0]
interval = 1
prev_sim_time = 0
sim_time = 0
sim_end = 86400*2

Chargers = []

C = Charger()
V = Vehicle()

# Input data 1
V.charging_efficiency = float(0.9)
V.battery_SOC = float(100)
V.mileage_efficiency = float(3.846)
# V.maximum_charge_rate = float(11500)
C.maximum_load = float(11500)
arrival_at_work = 1330
duration_at_work = 8400
arrival_at_home = 1620
mileage_classification = 220
travel_distance = 44
# Parameter Calculation
V.battery_size = float(mileage_classification / V.mileage_efficiency)
# C.maximum_load = float(V.maximum_charge_rate / V.charging_efficiency)
V.maximum_charge_rate = float(C.maximum_load*V.charging_efficiency)
V.update_capacity()
V.work_start = int(arrival_at_work) // 100
V.work_start += (( int(arrival_at_work) % 100 ) / 60)
V.work_duration = float(duration_at_work) / 3600
home_arrival = int(arrival_at_home) // 100
home_arrival += (( int(arrival_at_home) % 100 ) / 60)
V.commute_duration = round(((home_arrival - V.work_start - V.work_duration) * 3600), -1)
V.commute_distance = float(travel_distance) / 2
V.set_day_schedule(sim_end)
V.next_state_change = V.schedule[1][0]
V.update_log()
C.add_vehicle(V)
Chargers.append(C)


C = Charger()
V = Vehicle()
# Input data 2
V.charging_efficiency = float(0.9)
V.battery_SOC = float(50)
V.mileage_efficiency = float(3.571)
# V.maximum_charge_rate = float(3300)
C.maximum_load = float(3300)
arrival_at_work = 1130
duration_at_work = 1
arrival_at_home = 1200
mileage_classification = 238
travel_distance = 47.6
# Parameter Calculation
V.battery_size = float(mileage_classification / V.mileage_efficiency)
# C.maximum_load = float(V.maximum_charge_rate / V.charging_efficiency)
V.maximum_charge_rate = float(C.maximum_load*V.charging_efficiency)
V.update_capacity()
V.work_start = int(arrival_at_work) // 100
V.work_start += (( int(arrival_at_work) % 100 ) / 60)
V.work_duration = float(duration_at_work) / 3600
home_arrival = int(arrival_at_home) // 100
home_arrival += (( int(arrival_at_home) % 100 ) / 60)
V.commute_duration = round(((home_arrival - V.work_start - V.work_duration) * 3600), -1)
V.commute_distance = float(travel_distance) / 2
V.set_day_schedule(sim_end)
V.next_state_change = V.schedule[1][0]
V.update_log()
C.add_vehicle(V)
Chargers.append(C)

for C in Chargers:
    sim_time = 0
    A = C.current_vehicle
    while sim_time <= sim_end:
        # Update vehicle for previous interval
        current_interval = sim_time - prev_sim_time
        if current_interval > 0:
            if A.location == "HOME":
                if C.occupied:
                    C.charge(current_interval)
            if ((A.location == "DRIVING_WORK") or (A.location == "DRIVING_HOME")):
                A.battery_capacity -= (((A.commute_distance/A.commute_duration) / A.mileage_efficiency) * current_interval)
                A.update_SOC()
                A.current_time = sim_time
                A.update_log()
                C.current_time = A.current_time
                C.update_load()
            if A.location == "WORK":
                A.current_time += current_interval
                C.current_time = A.current_time
                C.update_load()
        
        # Check for loaction change
        for x in A.schedule:
            if x[0] == sim_time:
                # update next_state_change
                if A.schedule.index(x) < (len(A.schedule) - 1):
                    A.next_state_change = A.schedule[1 + A.schedule.index(x)][0]
                else:
                    A.next_state_change = sim_end
                # No location Change
                if A.location == x[1]:
                    A.location = x[1]
                # Location change
                else:
                    A.location = x[1]
                    if x[1] == "HOME":
                        C.add_vehicle(A)
                    if x[1] == "DRIVING_WORK":
                        if C.occupied:
                            C.remove_vehicle()
        
        # Update next sim time
        prev_sim_time = sim_time
        next_interval_time = (math.floor(prev_sim_time / interval) + 1) * interval
        sim_time = min(next_interval_time,A.next_state_change,sim_end)
        if prev_sim_time == sim_end:
            sim_time += 1

plot_time, plot_load = agregate_loads(Chargers, sim_end, interval)
plot_time = np.array(plot_time)
# plot_time = plot_time / 3600
plot_load = np.array(plot_load)
# plot_load = plot_load * 0.9

time_diff, load_diff = difference_loads(plot_time,plot_time_d,plot_load,plot_charge_d,sim_end,interval)

plt.plot(plot_time,plot_load)
plt.plot(plot_time_d,plot_charge_d)
labels = ['Python load','Gridlab-D Load']
plt.legend(labels)
plt.xlabel("Time (sec)")
plt.ylabel("Combined Charging Rate (Watts)")
plt.show()

# plt.plot(time_diff,load_diff)
# plt.xlabel("Time (sec)")
# plt.ylabel("Python - Gridlab-d Loads (Watts)")
# plt.show()

energy_input = numerical_integration(plot_time, plot_load)
energy_input = (energy_input / 3600) / 1000                         # W*s -> kW * h

energy_input_d = numerical_integration(plot_time_d, plot_charge_d)
energy_input_d = (energy_input_d / 3600) / 1000

avg_dist_actual = 0.0
for C in Chargers:
    avg_dist_actual += 2*C.current_vehicle.commute_distance
avg_dist_actual = avg_dist_actual / len(Chargers)

energy_error = energy_input_d - energy_input
pct_energy_error = (energy_error / energy_input) * 100
