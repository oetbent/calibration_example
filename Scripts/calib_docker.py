from bayes_opt import BayesianOptimization
import subprocess
import json
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--file', default= "config.json", help='config file to modify including path')
parser.add_argument('--sfile', default= "config_mod.json", help='config file to save to including path')
parser.add_argument('--dur', default= 3650, help='simulation duration')
parser.add_argument('--demo', default="Garki_single_demographics.json", help='demographic file to reference')
parser.add_argument('--output', default="output/InsetChart.json", help='path to output file')
parser.add_argument('--data', default="data/time_series_19-covid-Confirmed.csv")
args = parser.parse_args()
file, sfile, demo, dur, output, data = args.file, args.sfile, args.demo, args.dur, args.output, args.data

population = 10000000 #this needs to be taken from Demographic file
#Demographics file seeds the infection?
#campaign file contains this

# specify paths
binary_path = "/model/IDM/EMOD/build/x64/Release/Eradication/Eradication"

START_DATE = {
  'Italy': '1/31/20'
}

i = 0

def write_params(r,inc,inf):
	### write parameters to config file
	config_file = open( file )
	config_json = json.load( config_file )
	config_file.close()
	# insert parameter values into config file
	# modify parameter values, e.g. "base_infectivity" my understanding R0/infectious_period
	config_json["parameters"]["Base_Infectivity"] = r
	#incubation period 1/a
	config_json["parameters"]["Incubation_Period_Constant"] = inc
	#infectious period
	config_json["parameters"]["Infectious_Period_Constant"] = inf
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
	country_df = df[df['Country/Region'] == country]
	return country_df.iloc[0].loc[START_DATE[country]:]


def epi_function(r, inc, inf):
	write_params(r,inc,inf)
	global i
	run(i)

	##
	model_infected = outputs(i,'Infected')*population
	real_infected = load_confirmed('Italy')

	### RMSE
	real = np.array(real_infected.tolist())
	model = np.array(model_infected.tolist())[:real.shape[0], ]
	i+=1


	return -np.sum(np.sqrt((real-model)**2))


# Bounded region of parameter space
pbounds = {'r': (0.1, 1), 'inc': (1, 10), 'inf': (1,12)}

optimizer = BayesianOptimization(
    f=epi_function,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=2,
    n_iter=3,
)

print(optimizer.max)
