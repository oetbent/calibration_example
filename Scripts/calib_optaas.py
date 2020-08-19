from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import subprocess
import json
import argparse
import pandas as pd
import numpy as np

from mindfoundry.optaas.client.client import OPTaaSClient
from mindfoundry.optaas.client.goal import Goal
# from mindfoundry.optaas.client.expressions import Constraint
from mindfoundry.optaas.client.parameter import IntParameter, FloatParameter, CategoricalParameter, BoolParameter, ChoiceParameter, GroupParameter, SubsetParameter

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

population = 10000 #this needs to be taken from Demographic file
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
	config_json["parameters"]["Base_Infectious_Period"] = inf
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


# def epi_function(pin, r, inc, inf):
# 	write_params(pin, r,inc,inf)
# 	global i
# 	run(i)

# 	##
# 	model_infected = outputs(i,'Infected')
# 	# print('model_infected', model_infected)
# 	real_infected = load_confirmed(country)

# 	### RMSE
# 	real = np.array(real_infected["mean"].tolist())
# 	print('real', real)
# 	model = np.array(model_infected.tolist()).reshape(10,365).mean(axis=1)
# 	print('model', model)
# 	i+=1


# 	return -np.sum(np.sqrt((real-model)**2))

client = OPTaaSClient('https://edu.optaas.mindfoundry.ai', 'laraaG8eicaeCahxeiy2')

# Define your parameters, e.g.
pin = FloatParameter('pin', minimum=0, maximum=1) 
r = FloatParameter('r', minimum=0, maximum=2) 
inc = IntParameter('inc', minimum=0, maximum = 20)
inf = IntParameter('inf', minimum=0, maximum=20) 

parameters = [
    pin,
    r,
    inc,
    inf
]


def scoring_function(pin, r, inc, inf):
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

	#RMS Error
	score = np.sum(np.sqrt((real-model)**2))

	print('score', score)

	return score  # You can return just a score, or a tuple of (score, variance)

# Bounded region of parameter space
# Create your task
task = client.create_task(
    title='EMOD calib',
    parameters=parameters,
    # constraints=constraints,
    goal=Goal.min  # or Goal.max as appropriate
    # min_known_score= 11750#, max_known_score=44  # optional
)

# Run your task
best_result = task.run(
    scoring_function,
    max_iterations=5
    # score_threshold=32  # optional (defaults to the max_known_score defined above since the goal is "max")
)

print("Best Result: ", best_result)