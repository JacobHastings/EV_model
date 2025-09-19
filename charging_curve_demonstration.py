import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd
import numpy as np
import json
import glmanip

from Vehicle_class import Vehicle
from Charger_class import Charger

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
interval = 1
sim_end = 40000 #86400*1
prev_sim_time = 0
P_rate = 1
P_plot = []
SOC_plot = []

for i in [0,1,2,3]:
    P_rate = i+1
    P_plot.append([])
    SOC_plot.append([])
    C = Charger()
    V = Vehicle()
    
    # Input data 1
    V.charging_efficiency = float(1)
    V.battery_SOC = float(0)
    V.mileage_efficiency = float(3.846)
    # V.maximum_charge_rate = float(11500)
    
    arrival_at_work = 1330
    duration_at_work = 8400
    arrival_at_home = 1620
    mileage_classification = 220
    travel_distance = 44
    # Parameter Calculation
    V.battery_size = float(50)
    C.maximum_load = float(V.battery_size*P_rate*10000)
    # C.maximum_load = float(V.maximum_charge_rate / V.charging_efficiency)
    V.maximum_charge_rate = float(V.battery_size*P_rate*1000)
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
    C.DC = True
    if i==3:
        # C.maximum_load = float(.32825*V.battery_size*1000)
        # V.maximum_charge_rate = float(C.maximum_load)
        V.maximum_charge_rate = float(.2*V.battery_size*1000)
        C.maximum_load = float(V.maximum_charge_rate*2)
        C.DC = False
    
    
    
    sim_time = 0
    A = C.current_vehicle
    while A.battery_SOC < 100:
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
        
        # Update Plots
        P_plot[i].append(C.current_charging_rate/V.battery_size/1000)
        SOC_plot[i].append(V.battery_SOC)
    
    del SOC_plot[i][0]
    del P_plot[i][0]
    plt.plot(SOC_plot[i],P_plot[i])


plt.xlabel("Battery SOC")
plt.ylabel("P")
plt.ylim(-0.2,4)
labels = ['C rate - 1','C rate - 2','C rate - 3','AC Charging']
plt.legend(labels)
plt.grid()
plt.show()
# Chargers = []
# Chargers.append(C)
# plot_time, plot_load = agregate_loads(Chargers, sim_time+100, interval)
# plot_time = np.array(plot_time)
# plot_time = np.delete(plot_time, 0)
# # plot_time = plot_time / 3600
# plot_load = np.array(plot_load)
# plot_load = np.delete(plot_load, 0)
# # plot_load = plot_load * 0.9

# plt.plot(plot_time,plot_load)
# plt.xlabel("Time (sec)")
# plt.ylabel("Combined Charging Rate (Watts)")
# plt.show()