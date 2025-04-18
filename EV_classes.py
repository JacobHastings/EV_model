# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd
import numpy as np
import json
import glmanip



def clean_timestamp(dataframe):
    dataframe['# timestamp'] = [datetime.datetime.strptime(date.split(' PDT')[0], "%Y-%m-%d %H:%M:%S") for date in dataframe['# timestamp']]
    
def clean_timestamp_EST(dataframe):
    dataframe['# timestamp'] = [datetime.datetime.strptime(date.split(' EST')[0], "%Y-%m-%d %H:%M:%S") for date in dataframe['# timestamp']]


def output_from_gridlabd():
    filename = "C:/Users/jacob/Documents/MatpowerWrapper/EVtest/driverfile.csv"
    raw_data_charger1 = pd.read_csv(filename, skiprows=8, sep=',')
    clean_timestamp(raw_data_charger1)
    filename = "C:/Users/jacob/Documents/MatpowerWrapper/EVtest/driverfile3.csv"
    raw_data_charger2 = pd.read_csv(filename, skiprows=8, sep=',')
    clean_timestamp(raw_data_charger2)
    filename = "C:/Users/jacob/Documents/MatpowerWrapper/EVtest/driverfile2.csv"
    raw_data_house = pd.read_csv(filename, skiprows=8, sep=',')
    clean_timestamp(raw_data_house)
    
    time = raw_data_house['# timestamp']
    house_load = raw_data_house['total_load']
    EV1_SOC = raw_data_charger1[' battery_SOC']
    EV1_charging = raw_data_charger1[' actual_charge_rate']
    EV2_SOC = raw_data_charger2[' battery_SOC']
    EV2_charging = raw_data_charger2[' actual_charge_rate']
    combined_charging = EV1_charging #+ EV2_charging
    combined_charging = combined_charging
    
    # Convert timestamps into seconds since start of simulation
    plot_time = (np.array(time)).astype(int) / 10**9    #default is nanoseconds
    plot_time = plot_time - plot_time[0]
    
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
    
    

class Charger:
    # Class variables (shared by all in class)
    
    def __init__(self):
        # Variables unique to each instance *Be sure to update info()*
        self.occupied = False
        self.current_charging_rate = 0
        self.load = 0
        self.current_vehicle = Vehicle()
        self.maximum_load = 1700
        self.current_time = 0
        self.load_log = [(0.0,0.0)]
        self.name = 'NONE'
        
        
    def add_vehicle(self,EV):
        # default for now
        self.current_charging_rate = 0
        if self.occupied:
            print("Cannot add vehicle. Charger is currently occupied by vehicle {}.".format(self.current_vehicle.index))
        else:
            self.occupied = True
            self.current_vehicle = EV
            
    def remove_vehicle(self):
        if self.occupied:
            self.occupied = False
            self.current_charging_rate = 0
            self.current_time = self.current_vehicle.current_time
            self.update_load()
            return self.current_vehicle
        else:
            print("No vehicle to remove")
            
    def update_load(self):
        if self.occupied:
            self.load = self.current_charging_rate / self.current_vehicle.charging_efficiency
        else:
            self.load = 0
        
        if self.current_time == self.load_log[len(self.load_log)-1][0]:
            self.load_log[len(self.load_log)-1] = (self.current_time,self.load)
        else:
            self.load_log.append((self.current_time,self.load))
        
    def charge(self,time):      # time is in seconds
        # Check for vehicle presence
        if not self.occupied:
            print("No vehicle to charge")
            return
        # Disconnect already full vehicle
        if self.current_vehicle.battery_SOC >= 100:
            # Update vehicle information
            self.current_vehicle.update_SOC()
            self.current_vehicle.current_time += time
            self.current_vehicle.update_log()
            
            # Update charger information
            self.current_time = self.current_vehicle.current_time
            self.remove_vehicle()
            self.update_load()
            return
        
        # Set/check charging rate (nonzero/negative, doesn't exceed vehicle parameters, doesn't exceed charger parameters)
        if (self.current_charging_rate <= 0) or (self.current_charging_rate > self.current_vehicle.maximum_charge_rate) or (self.current_charging_rate > (self.maximum_load * self.current_vehicle.charging_efficiency)):
            # Default to maximum if not set
            self.current_charging_rate = min(self.current_vehicle.maximum_charge_rate, (self.maximum_load * self.current_vehicle.charging_efficiency))
        
        # Charge battery
        # Check for overcharge
        if (self.current_vehicle.battery_capacity + ((self.current_charging_rate * time / 3600) / 1000)) > self.current_vehicle.battery_size:
            # Charge up to full over given interval
            remaining_charge = self.current_vehicle.battery_size - self.current_vehicle.battery_capacity #remaining charge in kWh
            self.current_charging_rate = remaining_charge / (time/3600)
            self.current_vehicle.battery_capacity = self.current_vehicle.battery_size
        # Charge at current rate
        else:
            self.current_vehicle.battery_capacity += ((self.current_charging_rate * time / 3600) / 1000)
        
        # Update vehicle information
        self.current_vehicle.update_SOC()
        self.current_vehicle.current_time += time
        self.current_vehicle.update_log()
        
        # Update charger information
        self.current_time = self.current_vehicle.current_time
        self.update_load()
        
        
        
            
            
        
    def info(self):
        help_string = "------ Charger class variables -----\n"
        help_string = help_string + "occupied: \t\t\t\tboolean indicating the presence of a vehicle in the charger \n"
        help_string = help_string + "current_charging_rate: \tthe rate of energy being delivered to the battery accounting for charging efficiency in Watts \n"
        help_string = help_string + "load: \t\t\t\t\tthe load drawn by the charger in Watts \n"
        help_string = help_string + "current_vehicle: \t\tthe most recent vehicle within the charger\n"
        help_string = help_string + "maximum_load: \t\t\tthe highest output of this charger in Watts\n"
        help_string = help_string + "current_time: \t\t\tseconds that have passed since the start of the simulation\n"
        help_string = help_string + "load_log: \t\t\t\tlist of pairs of with (current_time,load)\n"
        
        
        
        help_string = help_string + "\n------ Charger class functions -----\n"
        help_string = help_string + "add_vehicle(EV): \t\ttakes in a Vehicle object (EV) and adds it to the charger if it is unoccupied\n"
        help_string = help_string + "remove_vehicle(): \t\tremoves the current vehicle and returns the Vehicle object\n"
        help_string = help_string + "update_load(): \t\t\tchanges load based on current_charging_rate or sets to 0 if unoccupied; adds to load_log\n"
        help_string = help_string + "charge(time): \t\t\ttakes in an amount of time in seconds (time) and charges the current vehicle according to the defined parameters\n"
        
        print(help_string)
        
class Vehicle:
    
    # Class variables (shared by all in class)
    
    def __init__(self):
        # Variables unique to each instance *Be sure to update info()*
        self.battery_capacity = 0
        self.battery_SOC = 0.0
        self.battery_size = 20
        self.maximum_charge_rate = 1700
        self.charging_efficiency = 0.9
        self.index = -1             # -1 for unassigned
        
        self.current_time = 0
        self.commute_duration = 1800
        self.commute_distance = 15.0
        self.work_start = 5
        self.work_duration = 11.5
        self.mileage_efficiency = 3.846     # default 0.26kWh/mile
        self.next_state_change = 0
        self.location = "HOME"
        self.SOC_log = [(0.0,0.0)]
        self.schedule = [(0.0,"HOME")]
        
        
    def update_SOC(self):
        self.battery_SOC = (self.battery_capacity / self.battery_size) * 100
        
    def update_capacity(self):
        self.battery_capacity = (self.battery_SOC / 100) * self.battery_size
        
    def update_log(self):
        if self.current_time == self.SOC_log[len(self.SOC_log)-1][0]:  # Overwrite if current time is same as last entry
            self.SOC_log[len(self.SOC_log)-1] = (self.current_time,self.battery_SOC)
        else:
            self.SOC_log.append((self.current_time,self.battery_SOC))
            
    def set_day_schedule(self,sim_end):
        day = 0
        schedule_time = 0
        self.schedule = [(schedule_time,"HOME")]                                    # Assume sim starts at home for now
        days = math.ceil(sim_end/86400)                                             # Determine how many days to schedule
        while day < days:
            schedule_time = 86400*day                                               # Start at 0:00 for that day
            schedule_time += ((self.work_start * 3600) - self.commute_duration)
            self.schedule.append((schedule_time,"DRIVING_WORK"))
            schedule_time += self.commute_duration
            self.schedule.append((schedule_time,"WORK"))
            schedule_time += (self.work_duration*3600)
            self.schedule.append((schedule_time,"DRIVING_HOME"))
            schedule_time += self.commute_duration
            self.schedule.append((schedule_time,"HOME"))
            day += 1
            
        self.next_state_change = self.schedule[1][0]                                # Assume leaving home is first action
        

        
        
    def info(self):
        help_string = "------ Vehicle class variables -----\n"
        help_string = help_string + "battery_capacity: \t\tcurrent energy stored in the battery in kWh \n"
        help_string = help_string + "battery_SOC: \t\t\tState of charge of the battery as a percentage \n"
        help_string = help_string + "battery_size: \t\t\tmaximum energy stored in the battery in kWh \n"
        help_string = help_string + "maximum_charge_rate:\tthe highest allowed charge rate in Watts \n"
        help_string = help_string + "charging_efficiency: \tcharging efficiency of the vehicle expressed as a decimal \n"
        help_string = help_string + "index: \t\t\t\t\tan integer assigned to this vehicle for a label (-1 indicates it is unset)\n"
        
        help_string = help_string + "current_time: \t\t\tseconds that have passed since the start of the simulation\n"
        help_string = help_string + "commute_duration: \t\ttime to drive between work and home in seconds\n"
        help_string = help_string + "commute_distance: \t\tdistance of drive between work and home in miles\n"
        help_string = help_string + "work_start: \t\t\ttime of day to begin work (24 hour clock; 5 -> 5:00am, 14.5 -> 2:30pm) \n"
        help_string = help_string + "work_duration: \t\t\ttime spent at work in hours\n"
        help_string = help_string + "mileage_efficiency: \tefficiency of the vehicle in miles/kWh\n"
        help_string = help_string + "next_state_change: \t\ttime of the next scheduled change in seconds from start of simulation\n"
        help_string = help_string + "location: \t\t\t\tstring indicating location (HOME,WORK,DRIVING_HOME,DRIVING_WORK)\n"
        help_string = help_string + "SOC_log: \t\t\t\tlist of pairs of with (current_time,battery_SOC)\n"
        help_string = help_string + "schedule: \t\t\t\tlist of pairs of with (time,location) for the day\n"
        
        
        
        help_string = help_string + "\n------ Vehicle class functions -----\n"
        help_string = help_string + "update_SOC(): \t\t\tChanges battery_SOC to match battery_capacity \n"
        help_string = help_string + "update_capacity(): \t\tChanges battery_capacity to match battery_SOC \n"
        help_string = help_string + "update_log(): \t\t\tAdds or alters SOC_log based on current_time and battery_SOC\n"
        help_string = help_string + "set_day_schedule(): \tsets schedule for the current day\n"
        
        
        print(help_string)
        
        
        
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
        
##############################################################################
#################################### Main ####################################
##############################################################################

time = [0]
load = [0.0]
interval = 300
prev_sim_time = 0
sim_time = 0
sim_end = 86400*7

A = Vehicle()
A.index = 1
A.battery_size = 8.58034
A.battery_SOC = 60.8696
A.update_capacity()
A.commute_distance = 20 / 2         # Gridlab-D lists round trip distance
A.update_log()

B = Vehicle()
B.index = 2
B.battery_size = 25
B.battery_SOC = 30
B.update_capacity()

#C = Charger()
Chargers = []

plot_time_d, plot_charge_d = output_from_gridlabd_v2()

basedir = ""
dir_for_glm ='test.glm'
glm_lines = glmanip.read(dir_for_glm,basedir,buf=[])
[model,clock,directives,modules,classes] = glmanip.parse(glm_lines)

################## Read in Vehicles #######################
file = open("C:/Users/jacob/Documents/MatpowerWrapper/EVtest/EV_dict/Substation_2_glm_dict.json")
EV_dict_raw = json.load(file)
EV_dict = EV_dict_raw['ev']

EV_dict = model['evcharger_det']

EV_index = 0
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
    V.maximum_charge_rate = float(EV_dict[i]['maximum_charge_rate'])
    
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
    V.commute_duration = round(((home_arrival - V.work_start - V.work_duration) * 3600), -1)    # Duration is in seconds
    # V.commute_distance = EV_dict[i]['daily_miles'] / 2
    V.commute_distance = float(EV_dict[i]['travel_distance']) / 2
    
    V.set_day_schedule(sim_end)
    V.next_state_change = V.schedule[1][0]
    V.update_log()
    
    C.add_vehicle(V)
    Chargers.append(C)



# (Vehicle, Charging time, Charging rate)
# Vehicle_queue = [(B,4000,1200),(A,10000,0)]

# while len(Vehicle_queue) > 0:
#     C.add_vehicle(Vehicle_queue[0][0])
#     C.current_charging_rate = Vehicle_queue[0][2]
#     print("Vehicle number: {}\nStarting SOC: {}%\nCharge time: {} seconds".format(C.current_vehicle.index,C.current_vehicle.battery_SOC,Vehicle_queue[0][1]))
#     C.charge(Vehicle_queue[0][1])
#     time.append(time[len(time)-1]+Vehicle_queue[0][1])
#     load.append(C.load)
#     print("Ending SOC: {}%\nLoad: {} Watts\n".format(C.current_vehicle.battery_SOC,C.load))
#     C.remove_vehicle()
#     del Vehicle_queue[0]


# C.add_vehicle(A)
# print("Battery at {}%".format(C.current_vehicle.battery_SOC))

# while(C.current_vehicle.battery_SOC < 100):
#     print("Charging 1 hour")
#     C.charge(3600)
#     time.append(time[len(time)-1]+3600)
#     load.append(C.load)
#     print("Charging rate: {} Watts".format(C.current_charging_rate))
#     print("Load: {} Watts".format(C.load))
#     print("Battery at {}%".format(C.current_vehicle.battery_SOC))
    
# C.remove_vehicle()

# plt.plot(time,load)
# plt.show()

# C.add_vehicle(A)
# A.set_day_schedule(sim_end)
# A.next_state_change = A.schedule[1][0]


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
plot_load = plot_load * 0.9
# plot_load = plot_load / 1000

# plot_time = []
# plot_load = []
# load_report = []
# for x in C.load_log:
#     plot_time.append(x[0])
#     plot_load.append(x[1])
#     load_report.append((str(datetime.timedelta(seconds=x[0])),x[1]))
    
# plot_time_d, plot_charge_d, plot_house_d = output_from_gridlabd()


# #plot_diff = plot_charge_d - plot_load
    
plt.plot(plot_time,plot_load)
plt.plot(plot_time_d,plot_charge_d)
labels = ['Python load','Gridlab-D Load']
plt.legend(labels)
plt.show()


energy_input = numerical_integration(plot_time, plot_load)
energy_input = (energy_input / 3600) / 1000                         # W*s -> kW * h
avg_distance_per_car = (energy_input * 3.846) / 100               # 3.846 mi/kWhr, 100 vehicles
avg_distance_per_car_per_day = avg_distance_per_car / 7             # simulation for a week

energy_input_d = numerical_integration(plot_time_d, plot_charge_d)
energy_input_d = (energy_input_d / 3600) / 1000
avg_distance_per_car_d = (energy_input_d * 3.846) / 100
avg_distance_per_car_per_day_d = avg_distance_per_car_d / 7

avg_dist_actual = 0.0
for C in Chargers:
    avg_dist_actual += 2*C.current_vehicle.commute_distance
avg_dist_actual = avg_dist_actual / 100
