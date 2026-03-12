import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd
import numpy as np
import json
import glmanip
import copy

from Vehicle_class import Vehicle
from Charger_class import Charger

################## Define shared variables #######################
broker_port = '60000'
sim_interval = '300'
##################################################################

dir_for_glm = "Substation_2.glm"

glm_lines = glmanip.read(dir_for_glm,"",buf=[])
[model,clock,directives,modules,classes] = glmanip.parse(glm_lines)

################## Read in Vehicles #######################
EV_dict = model['evcharger_det']

helics_config_py = {}
helics_config_py['name'] = 'py'
helics_config_py['coreName'] = 'py'
helics_config_py['loglevel'] = 'warning'
helics_config_py['coreType'] = 'zmq'
helics_config_py['period'] = '1'
helics_config_py['broker_port'] = broker_port
helics_config_py['publications'] = [];
helics_config_py['subscriptions'] = [];
helics_config_gld = copy.deepcopy(helics_config_py)
helics_config_py['timeDelta'] = '0'
helics_config_gld['name'] = 'gld'
helics_config_gld['coreName'] = 'gld'
helics_config_gld['only_transmit_on_change'] = 'false'

model_new = copy.deepcopy(model)
# Hardcoded temp fix to remove building EVs attached to non-triplex meters
del model_new['evcharger_det']['meter_bldg_28_ev']
# Create triplex loads to replace evcharger objects
for EV_name in model_new['evcharger_det']:
    model_new['triplex_load'][EV_name] = {}
    # glmanip does the name automatically
    house_name = model['evcharger_det'][EV_name]['parent']
    meter_name = model['house'][house_name]['parent']
    # model_new['triplex_load'][EV_name]['parent'] = meter_name
    model_new['triplex_load'][EV_name]['phases'] = model['triplex_meter'][meter_name]['phases']
    model_new['triplex_load'][EV_name]['base_power_12'] = '10'
    model_new['triplex_load'][EV_name]['power_fraction_12'] = '1'
    model_new['triplex_load'][EV_name]['power_pf_12'] = '1'
    # Add in intermediary triplex meters
    new_meter_name = EV_name + "_mev"
    model_new['triplex_meter'][new_meter_name] = {}
    # model_new['triplex_meter'][new_meter_name]['name'] = new_meter_name # Check if needed (not needed)
    model_new['triplex_meter'][new_meter_name]['parent'] = meter_name
    model_new['triplex_meter'][new_meter_name]['groupid'] = 'TPX_EV_METERS'
    model_new['triplex_meter'][new_meter_name]['phases'] = model['triplex_meter'][meter_name]['phases']
    model_new['triplex_meter'][new_meter_name]['nominal_voltage'] = '120.0'
    # Add triplex load to new meter
    model_new['triplex_load'][EV_name]['parent'] = new_meter_name
    
    # Update helics config files
    # Charger load data
    helics_config_py['publications'].append({'key':str(EV_name)})
    helics_config_gld['subscriptions'].append({'key':str(helics_config_py['name'] + '/' + EV_name),
                                               'type':str('double'),
                                               'unit':str('W'),
                                               'info':{'object':str(EV_name),
                                                       'property':str('base_power_12')
                                                       }
                                               })
# Substation data
substation_key = 'distribution_load'
substation_name = 'network_node'
helics_config_gld['publications'].append({'key':str(substation_key),
                                           'type':str('complex'),
                                           'info':{'object':str(substation_name),
                                                   'property':str('distribution_load')
                                                   }
                                           })
helics_config_py['subscriptions'].append({'key':str(helics_config_gld['name'] + '/' + substation_key)})

# Delete previous evcharger objects
del model_new['evcharger_det']
del model_new['currdump']
del model_new['voltdump']
del model_new['group_recorder']
for key in model_new["helics_msg"]:
    model_new["helics_msg"][key]["configure"] = "helics_config_gld.json"

# Add in group recorder
model_new['group_recorder'] = {}
model_new['group_recorder']['EV_meter_recorder'] = {}
# model_new['group_recorder']['EV_meter_recorder']['name'] = 'EV_meter_recorder'
model_new['group_recorder']['EV_meter_recorder']['group'] = 'groupid=TPX_EV_METERS'
model_new['group_recorder']['EV_meter_recorder']['property'] = 'measured_real_power'
model_new['group_recorder']['EV_meter_recorder']['file'] = 'EV_charger_output.csv'
model_new['group_recorder']['EV_meter_recorder']['interval'] = sim_interval

# Set minimum timestep to sim_interval
timestep_str = '#set minimum_timestep=' + str(60) + ';\n'
for i, s in enumerate(directives):
    if '#set minimum_timestep=' in s:
        directives[i] = timestep_str

ofn = "Substation_2_mod.glm"
glmanip.write(ofn, model_new, clock, directives, modules, classes)
ofn = "helics_config_py.json"
out_file = open(ofn, "w")
json.dump(helics_config_py, out_file, indent=4)
out_file.close()
ofn = "helics_config_gld.json"
out_file = open(ofn, "w")
json.dump(helics_config_gld, out_file, indent=4)
out_file.close()