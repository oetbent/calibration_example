from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import subprocess
import json
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--file', default= "config_vector.json", help='config file to modify including path')
parser.add_argument('--sfile', default= "config_mod.json", help='config file to save to including path')
parser.add_argument('--dur', default= 3650, help='simulation duration')
parser.add_argument('--demo', default="mod_demographics.json", help='demographic file to reference')
parser.add_argument('--output', default="output/InsetChart.json", help='path to output file')
parser.add_argument('--data', default="data/draw1.csv")
parser.add_argument('--demo_file', default= "Demographics/Namawala_single_node_demographics.json", help='demographic file to modify including path')

args = parser.parse_args()
file, sfile, demo, dur, output, data, dfile= args.file, args.sfile, args.demo, args.dur, args.output, args.data, args.demo_file

population = 1000 #this needs to be taken from Demographic file
#Demographics file seeds the infection?
#campaign file contains this

# specify paths
binary_path = "/model/IDM/EMOD/build/x64/Release/Eradication/Eradication"

i = 0
country = 96 #Kampala 5 Amudat

def write_params(pin,r,inc,inf):
	### write parameters to config file

	intital = open( dfile )
	intial_prev = json.load(intital)
	intital.close()
	intial_prev["Nodes"][0]["IndividualAttributes"]["PrevalenceDistribution1"] = pin
	intial_prev["Nodes"][0]["IndividualAttributes"]["PrevalenceDistributionFlag"] = 0

	modified_dfile = open( "Demographics/"+demo, "w" )
	json.dump( intial_prev, modified_dfile, sort_keys=True, indent=4 )
	modified_dfile.close()

	config_file = open( file )
	config_json = json.load( config_file )
	config_file.close()
	# insert parameter values into config file
	# modify parameter values, e.g. "base_infectivity" my understanding R0/infectious_period
	config_json["parameters"]["Base_Infectivity"] = r
	#incubation period 1/a
	config_json["parameters"]["Base_Incubation_Period"] = inc
	#infectious period
	config_json["parameters"]["Bae_Infectious_Period"] = inf
	#####
	#demographic file
	config_json["parameters"]["Demographics_Filenames"] = [demo]
	#simulation duration
	config_json["parameters"]["Simulation_Duration"] = dur
	#####
	# write the modified config file
	modified_file = open( sfile, "w" )
	json.dump( config_json, modified_file, sort_keys=True, indent=4 )
	modified_file.close()

def run(num):
	#run config file
	subprocess.call( [binary_path, "-C", "config_mod.json", "-O", str(num), "-I", "/model/Demographics/"] )

def outputs(ind, label = "" ):
    with open( str(ind) + '/InsetChart.json' ) as ref_sim:
        ref_data = json.loads( ref_sim.read() )

    channels =  sorted(ref_data["Channels"])

    if label in channels:
        out = ref_data["Channels"][label]["Data"]
    else:
        print("invalid label :", label)
        sys.exit(0)

    return pd.Series(out)


def load_confirmed(country):
	"""
	Load confirmed cases downloaded from HDX
	"""
	df = pd.read_csv(data)
	country_df = df[df["district"] == country]
	pre_intervention = country_df[country_df["year"] < 1990]

	return pre_intervention


def epi_function(pin, r, inc, inf):
	write_params(pin, r,inc,inf)
	global i
	run(i)

	##
	model_infected = outputs(i,'Infected')
	# print('model_infected', model_infected)
	real_infected = load_confirmed(country)

	### RMSE
	real = np.array(real_infected["mean"].tolist())
	print('real', real)
	model = np.array(model_infected.tolist()).reshape(10,365).mean(axis=1)
	print('model', model)
	i+=1


	return -np.sum(np.sqrt((real-model)**2))


# Bounded region of parameter space
pbounds = {'pin': (0,1), 'r': (0, 2), 'inc': (0, 20), 'inf': (0,20)}

new_opt2  = BayesianOptimization(
    f=epi_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=40,
)
print(len(new_opt2.space))

# logger = JSONLogger(path="./logs.json")
# new_opt2.subscribe(Events.OPTIMIZATION_STEP, logger)

load_logs(new_opt2, logs=["./logs.json"])
print("New new_opt2 is now aware of {} points.".format(len(new_opt2.space)))


# new_opt2.probe(
# 	params = {'inc': 0.0, 'inf': 5.7356442634751925, 'pin': 1.0, 'r': 0.31836332930670513},
# 	lazy=True,
# 	)

# new_opt2.probe(
# 	params = {'inc': 2.0445771242785526, 'inf': 10.636516198747653, 'pin': 0.4417168510303957, 'r': 0.3460079168116166},
# 	lazy=True,
# 	)

# new_opt2.probe(
# 	params = {'inc': 2.5048201335585523, 'inf': 2.1312587884113614, 'pin': 0.25289106069855105, 'r': 0.3710532429053029},
# 	lazy=True,
# 	)


new_opt2.maximize(
    init_points=0,
    n_iter=17,
)

print('max', new_opt2.max)
print('res', new_opt2.res)
