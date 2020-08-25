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
parser.add_argument('--data', default="data/Monthly_EIR_data_for_Africa.csv")
parser.add_argument('--demo_file', default= "Demographics/Namawala_single_node_demographics.json", help='demographic file to modify including path')

args = parser.parse_args()
file, sfile, demo, dur, output, data, dfile= args.file, args.sfile, args.demo, args.dur, args.output, args.data, args.demo_file

population = 10000 #this needs to be taken from Demographic file
#Demographics file seeds the infection?
#campaign file contains this

# specify paths
binary_path = "/model/IDM/EMOD/build/x64/Release/Eradication/Eradication"

i = 0
country = "Uganda"

site = "Apac-Olami"
# site = "Arua-Cilio"
# site = "Kyenjojo_Kasiina"
# site = "Tororo-Namwaya"

def write_params(pin,r,inc,inf, trans):
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
	#Transmission Probability
	config_json["parameters"]["Vector_Species_Params"]["arabiensis"]["Transmission_Rate"] = trans
	config_json["parameters"]["Vector_Species_Params"]["funestus"]["Transmission_Rate"] = trans
	config_json["parameters"]["Vector_Species_Params"]["gambiae"]["Transmission_Rate"] = trans
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
	subprocess.call( [binary_path, "-C", "config_mod.json", "-O", country+"/"+site+"/"+str(num), "-I", "/model/Demographics/"] )

def outputs(ind, label = "" ):
    with open( country+'/'+site+'/'+str(ind) + '/InsetChart.json' ) as ref_sim:
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
	Load confirmed cases downloaded from monthly EIR
	"""
	df = pd.read_csv(data)
	country_df = df[df["Country"] == country]
	site_df = country_df[country_df["Site"] == site]
	start_month = int(site_df["Start Month"].tolist()[0])
	# pre_intervention = country_df[country_df["year"] < 1990]
	# print('site_df.head', site_df.head)
	EIR = []
	for i in range(12):
		x = site_df["value"+str(i+1)].tolist()
		EIR.append(x[0])


	if -9.0 in EIR:
		i = EIR.index(-9.0)
		EIR = np.asarray(EIR)
		EIR[i] = 'NaN'
	else:
		EIR = np.asarray(EIR)

	if start_month >1:
		real = np.append(EIR[start_month-1:],EIR[:-start_month-1])
	else:
		real = EIR
		# print('EIR', EIR)
		# print('start_month', start_month)
	return real


client = OPTaaSClient('https://edu.optaas.mindfoundry.ai', 'laraaG8eicaeCahxeiy2')

# Define your parameters, e.g.
pin = FloatParameter('pin', minimum=0, maximum=1) 
r = FloatParameter('r', minimum=0, maximum=2) 
inc = IntParameter('inc', minimum=0, maximum = 20)
inf = IntParameter('inf', minimum=0, maximum=20) 
trans = FloatParameter('trans', minimum = 0, maximum = 1)

parameters = [
    pin,
    r,
    inc,
    inf, 
    trans
]


def scoring_function(pin, r, inc, inf, trans):
	write_params(pin, r,inc,inf, trans)

	global i
	run(i)

	##
	model_infected = outputs(i,'Daily EIR')
	# print('model_infected', model_infected)
	real_infected = load_confirmed(country)

	### RMSE
	real = real_infected


	model = np.array(model_infected.tolist()).reshape(10,365)
	month = np.empty([5,12])
	for j in range(12):
		# print('model[:,j*30:(j+1)*30]', model[:,j*30:(j+1)*30])
		# print('np.sum(model[:,j*30:(j+1)*30], axis = 1) ', np.sum(model[:,j*30:(j+1)*30], axis = 1) )
		# print('np.sum(model[:,j*30:(j+1)*30], axis = 0) ', np.sum(model[:,j*30:(j+1)*30], axis = 0) )
		month[:,j] =np.sum(model[-5:,j*30:(j+1)*30], axis = 1) #last 5 years
	# print('model_year', model_year)
	# print('month', month)
	# model_month = model_year[:-5].reshape(12,30)
	model_med = np.median(month, axis = 0)
	# print('model[-1,:]', model[-1,:])
	# print('model_order', model_order)
	i+=1


	#RMS Error
	se = (real-model_med)**2
	score = np.mean(np.ma.masked_array(se, np.isnan(se)))

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
n = 1
# Run your task
best_result = task.run(
    scoring_function,
    max_iterations=n
    # score_threshold=32  # optional (defaults to the max_known_score defined above since the goal is "max")
)

print("Best Result: ", best_result)

m = 10

random_configs_values = [{'pin': np.random.uniform(0, 1), 
                          'r': np.random.uniform(0, 2),
                          'inc': np.random.randint(21),
                          'inf': np.random.randint(21),
                          'trans': np.random.uniform(0,1)} for _ in range(m)]

predictions = task.get_surrogate_predictions(random_configs_values)


from sklearn.decomposition.pca import PCA

surrogate_X = [[c['pin'], c['r'], c['inc'], c['inf'], c['trans']] for c in random_configs_values]
print('surrogate_X', surrogate_X)

pca = PCA(n_components=2)
surrogate_projected = pca.fit_transform(surrogate_X)
print('surrogate_projected', surrogate_projected)

mean = np.asarray([p.mean for p in predictions])
var = np.asarray([p.variance for p in predictions])
print('mean', mean)
print('var', var)


results = task.get_results()

evaluations_config_values = [r.configuration.values for r in results]
evaluations_score = [r.score for r in results]
print('evaluations_score', evaluations_score)
evaluations_X = [[c['pin'], c['r'], c['inc'], c['inf'], c['trans']] for c in evaluations_config_values]
print('evaluations_X', evaluations_X)
evaluations_projected = pca.transform(evaluations_X)
print('evaluations_projected', evaluations_projected)

samples = np.concatenate((surrogate_X.reshape(m,5),surrogate_projected.reshape(m,2),mean.reshape(m,1),var.reshape(m,1)))
evaluations = np.concatenate((evaluations_X.reshape(n,5),evaluations_projected.reshape(n,2),evaluations_score.reshape(n,1)))
np.save('surogate_results', samples)
np.save('task_results', evaluations)
