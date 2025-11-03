# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 16:23:04 2025

@author: Monish Mukherjee (monish.mukherjee@pnnl.gov)
Pacific Northwest National Laboratory
"""

import matplotlib.pyplot as plt
import math
import datetime
import pandas as pd
import numpy as np
import json
import glmanip
import copy
import os 

from Vehicle_class import Vehicle
from Charger_class import Charger

################## Define shared variables #######################
broker_port = '60000'
sim_interval = '300'
##################################################################



################## Read in Vehicles #######################
# file = open("C:/Users/jacob/Documents/MatpowerWrapper/EVtest/EV_dict/Substation_2_glm_dict.json")
# file = open("E:/Working_dir_Jacob/EV_dict/Substation_2_glm_dict.json")
# EV_dict_raw = json.load(file)
# EV_dict = EV_dict_raw['ev']

def update_GLD_EVs_with_Loads(model, helics_config_DSO_EV, include_helics = False):
    

    fed_name = helics_config_DSO_EV['name']
    EV_dict = model['evcharger_det']
    
    if include_helics:
        # helics_config_evsim_py = {}
        # helics_config_evsim_py['name'] = fed_name
        # helics_config_evsim_py['coreName'] = fed_name
        # helics_config_evsim_py['loglevel'] = 'warning'
        # helics_config_evsim_py['coreType'] = 'zmq'
        # helics_config_evsim_py['period'] = '1'
        # helics_config_evsim_py['broker_port'] = broker_port
        # helics_config_evsim_py['publications'] = [];
        # helics_config_evsim_py['subscriptions'] = [];
        # helics_config_evsim_py['timeDelta'] = '0'
        
        helics_config_gld = copy.deepcopy(helics_config_DSO_EV)
        helics_config_gld['name'] = 'gld'
        helics_config_gld['coreName'] = 'gld'
        # helics_config_gld['only_transmit_on_change'] = 'false'
        helics_config_gld['publications'] = [];
        helics_config_gld['subscriptions'] = [];
    
    model_new = copy.deepcopy(model)

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
        if include_helics:
            helics_config_DSO_EV['publications'].append({'key':str(EV_name)})
            helics_config_gld['subscriptions'].append({'key':str(helics_config_DSO_EV['name'] + '/' + EV_name),
                                                       'type':str('double'),
                                                       'unit':str('W'),
                                                       'info':{'object':str(EV_name),
                                                               'property':str('base_power_12')
                                                               }
                                                       })
    # Substation data
    if include_helics:
        substation_key = 'distribution_load'
        substation_name = 'network_node'
        helics_config_gld['publications'].append({'key':str(substation_key),
                                                   'type':str('complex'),
                                                   'info':{'object':str(substation_name),
                                                           'property':str('distribution_load')
                                                           }
                                                   })
        helics_config_DSO_EV['subscriptions'].append({'key':str(helics_config_gld['name'] + '/' + substation_key)})
    
    # Delete previous evcharger objects
    del model_new['evcharger_det']
    del model_new['currdump']
    del model_new['voltdump']
    del model_new['group_recorder']
    for key in model_new["helics_msg"]:
        model_new["helics_msg"][key]["configure"] = "helics_config_gld.json"
    
    # Delete Existing Recorders and Group Recorders and Add 
    model_new['group_recorder'] = {}
    model_new['group_recorder']['EV_meter_recorder'] = {}
    # model_new['group_recorder']['EV_meter_recorder']['name'] = 'EV_meter_recorder'
    model_new['group_recorder']['EV_meter_recorder']['group'] = 'groupid=TPX_EV_METERS'
    model_new['group_recorder']['EV_meter_recorder']['property'] = 'measured_real_power'
    model_new['group_recorder']['EV_meter_recorder']['file'] = 'outputs/EV_charger_output.csv'
    model_new['group_recorder']['EV_meter_recorder']['interval'] = sim_interval
    
    
    model_new['recorder'] = {}
    model_new['recorder']['Substation_recorder'] = {}
    model_new['recorder']['Substation_recorder']['parent'] = list(model_new['substation'].keys())[0]
    model_new['recorder']['Substation_recorder']['property'] = 'positive_sequence_voltage,distribution_load,distribution_power_A,distribution_power_B,distribution_power_C'
    model_new['recorder']['Substation_recorder']['file'] = 'outputs/Substation_output.csv'
    model_new['recorder']['Substation_recorder']['interval'] = sim_interval
    
    return model_new, helics_config_DSO_EV, helics_config_gld

if __name__ == "__main__":
    
    include_helics = True
    
    basedir = os.getcwd() + '\\' + 'distribution_simulation' + '\\'
    dir_for_glm = basedir + "Substation_2.glm"
    
    glm_lines = glmanip.read(dir_for_glm,"",buf=[])
    [model,clock,directives,modules,classes] = glmanip.parse(glm_lines)
    del model['evcharger_det']['meter_bldg_28_ev']
    
    
    fed_name = 'DSO_EV_sim'
    helics_config_DSO_EV = {}
    helics_config_DSO_EV['name'] = fed_name
    # helics_config_DSO_EV['coreInit'] = '--federates=1'
    helics_config_DSO_EV['coreName'] = fed_name + '_Federate'
    helics_config_DSO_EV['coreType'] = 'zmq'
    helics_config_DSO_EV['broker_port'] = str(broker_port)
    helics_config_DSO_EV['loglevel'] = 'warning'
    helics_config_DSO_EV['publications'] = [];
    helics_config_DSO_EV['subscriptions'] = [];
    
    model_new, helics_config_DSO_EV, helics_config_gld = update_GLD_EVs_with_Loads(model, helics_config_DSO_EV, include_helics)    
    
    ofn = dir_for_glm.replace('.glm', '_mod.glm')
    glmanip.write(ofn, model_new, clock, directives, modules, classes)
    
    if include_helics:
        with open(basedir + "helics_config_gld.json", 'w') as fp1:
            json.dump(helics_config_gld, fp1, indent=4)  
        fp1.close()
        
        helics_config_DSO_EV_filename = f"{fed_name}_helics_config.json"
        with open(helics_config_DSO_EV_filename, 'w') as fp2:
            json.dump(helics_config_DSO_EV, fp2, indent=4) 
        fp2.close()  
                
                
    
    
    
    