from Vehicle_class import Vehicle
from Charger_class import Charger
import matplotlib.pyplot as plt
import numpy as np

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

def update_time(Chargers, Vehicles, current_time):
    for C in Chargers:
        C.current_time = current_time
    for V in Vehicles:
        V.current_time = current_time


A = Vehicle()
B = Vehicle()
C = Charger()

A.maximum_charge_rate = 6700
A.charging_efficiency = 0.95
A.index = 0

B.maximum_charge_rate = 4000
B.charging_efficiency = 0.9
B.index = 1

C.maximum_load = 6000

sim_time = 0
Chargers = []
Chargers.append(C)
Vehicles = []
Vehicles.append(A)
Vehicles.append(B)

interval = 300

# Wait
sim_time += interval*12
update_time(Chargers, Vehicles, sim_time)


# First Vehicle arrives
C.add_vehicle(A)

# Charge 30 min
# for i in range(6):
#     C.charge(interval)
#     sim_time += interval
# Charge to full
while C.current_vehicle.battery_SOC <100:
    C.charge(interval)
    sim_time += interval

# First vehicle leaves
C.remove_vehicle()

# Wait
sim_time += interval*3
update_time(Chargers, Vehicles, sim_time)
#C.update_load()

# Second vehicle arrives
C.add_vehicle(B)

# Charge 60 min
for i in range(12):
    C.charge(interval)
    sim_time += interval
# Charge to full
# while C.current_vehicle.battery_SOC <100:
#     C.charge(interval)
#     sim_time += interval

# Sechond vehicle leaves
C.remove_vehicle()

# Wait
sim_time += interval*3
A.current_time = sim_time
B.current_time = sim_time
C.current_time = sim_time

# Plot Results
plot_time, plot_load = agregate_loads(Chargers, sim_time, interval)
plot_time = np.array(plot_time)
plot_load = np.array(plot_load)

plt.plot(plot_time/3600,plot_load)
plt.xlabel("Time (Hours)")
plt.ylabel("Load (Watts)")
plt.show()