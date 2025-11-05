# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 19:05:26 2025

@author: Monish Mukherjee (monish.mukherjee@pnnl.gov)
Pacific Northwest National Laboratory
"""

from Vehicle_class import Vehicle
from Charger_class import Charger
from Manager_class import Manager


vehicle_c_rating = 2.5

def intialize_EV_gld_info(EV_dict, sim_end):
    
    EV_index = 0
    Chargers = []
    for i in EV_dict:
        C = Charger()
        # C.name = EV_dict[i]['name']
        C.name = i
        # C.maximum_load = EV_dict[i]['max_charge'] / EV_dict[i]['efficiency']
        C.maximum_load = float(EV_dict[i]['maximum_charge_rate']) / float(EV_dict[i]['charging_efficiency'])
        V = Vehicle()
        
        V.battery_SOC = float(EV_dict[i]['battery_SOC'])
        # V.battery_size = float(EV_dict[i]['range_miles']) / float(EV_dict[i]['miles_per_kwh'])
        V.battery_size = float(EV_dict[i]['mileage_classification']) / float(EV_dict[i]['mileage_efficiency'])
        V.update_capacity()
        # V.charging_efficiency = float(EV_dict[i]['efficiency'])
        V.charging_efficiency = float(EV_dict[i]['charging_efficiency'])
        # V.mileage_efficiency = float(EV_dict[i]['miles_per_kwh'])
        V.mileage_efficiency = float(EV_dict[i]['mileage_efficiency'])
        # V.maximum_charge_rate = float(EV_dict[i]['max_charge'])
        # V.maximum_charge_rate = float(EV_dict[i]['maximum_charge_rate'])
        V.maximum_charge_rate = V.battery_size * 1000 * vehicle_c_rating   
        
        V.index = EV_index
        EV_index += 1
        
        # V.work_start = EV_dict[i]['arrival_work'] // 100            # hours
        V.work_start = int(EV_dict[i]['arrival_at_work']) // 100
        # V.work_start += (( EV_dict[i]['arrival_work'] % 100 ) / 60) # minutes
        V.work_start += (( int(EV_dict[i]['arrival_at_work']) % 100 ) / 60)
        # V.work_duration = EV_dict[i]['work_duration'] / 3600
        V.work_duration = float(EV_dict[i]['duration_at_work']) / 3600
        # home_arrival = EV_dict[i]['arrival_home'] // 100
        home_arrival = int(EV_dict[i]['arrival_at_home']) // 100
        # home_arrival += (( EV_dict[i]['arrival_home'] % 100 ) / 60)
        home_arrival += (( int(EV_dict[i]['arrival_at_home']) % 100 ) / 60)
        if home_arrival < V.work_start:
            home_arrival += 24

        V.commute_duration = round(((home_arrival - V.work_start - V.work_duration) * 3600), -1)    # Duration is in seconds
        # V.commute_distance = EV_dict[i]['daily_miles'] / 2
        V.commute_distance = float(EV_dict[i]['travel_distance']) / 2
        
        V.set_day_schedule(int(sim_end))
        V.next_state_change = V.schedule[1][0]
        V.update_log()
        
        C.add_vehicle(V)
        Chargers.append(C)
        
    return Chargers


def intialize_EV_Manager(Chargers, work_chargers_count):
    M = Manager()
    for i in range(work_chargers_count):
        C = Charger()
        C.maximum_load = 150000 # 150kw max
        # C.maximum_load = 0
        C.DC = True
        M.chargers.append(C)
        
    return M
    
    

def simulate_EVs(Chargers, M, sim_time, current_interval, sim_end):
    
    next_EV_update_time = sim_end
    if current_interval > 0:
        print('Simulating EVs .....')
        for C in Chargers:
            A = C.current_vehicle
            # Update vehicle for previous interval
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
            
            # print(C.load_log)
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
                        if x[1] == "WORK":
                            M.to_charge.append(A)
                        if x[1] == "DRIVING_HOME":
                            # remove from queue
                            if A in M.to_charge:
                                M.to_charge.remove(A)
                            # remove from work charger
                            for WC in M.chargers:
                                if ((WC.occupied) and (WC.current_vehicle == A)):
                                    WC.remove_vehicle()
            next_EV_update_time = min(next_EV_update_time, C.current_vehicle.next_state_change)
        # work charger
        M.current_time = sim_time
        M.simulate(current_interval)

    return Chargers, M, next_EV_update_time


def average_load_interval(load_log, sim_time, sim_interval):
    avg_out = 0
    # Generate list of all changes over last interval
    short_log = [point for point in load_log if (sim_time-sim_interval) <= point[0] < sim_time]
    # If no changes in last interval, return previous setting
    if len(short_log) < 1:
        for i in range(len(load_log)):
            if load_log[i][0] < sim_time:
                prev_i = i
            else:
                break
        return load_log[prev_i][1]
    
    # If there are changes in the last interval, calculate weighted avg
    for i in range(len(short_log)):
        # First entry not at beginning of interval
        if i==0 and (short_log[i][0] > (sim_time - sim_interval)):
            # Determine previous setting
            for j in range(len(load_log)):
                if load_log[j][0] < sim_time:
                    prev_j = j
                else:
                    break
            avg_out += load_log[prev_j][1] * (short_log[i][0] - (sim_time-sim_interval))
        # Last entry
        if i==(len(short_log)-1):
            avg_out += short_log[i][1] * (sim_time - short_log[i][0])
        # Normal entry
        else:
            avg_out += short_log[i][1] * (short_log[i+1][0] - short_log[i][0])
    # Normalize        
    avg_out = avg_out / sim_interval
    
    return avg_out


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


def combine_loads(time1,load1,time2,load2):
    combined_load = []
    time = []
    i = 0
    j = 0
    while i <= len(time1)-1 and j <= len(time2)-1:
        t = min(time1[i],time2[j])
        time.append(t)
        if t == time1[i] and t == time2[j]:
            combined_load.append(load1[i]+load2[j])
            i += 1
            j += 1
        else:
            if t == time1[i]:
                combined_load.append(load1[i]+load2[j-1])
                i += 1
            else:
                combined_load.append(load1[i-1]+load2[j])
                j += 1
    return time, combined_load
        
        
        
        