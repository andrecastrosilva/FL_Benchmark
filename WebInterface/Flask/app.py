# Modules:
from prometheus_client import CollectorRegistry, Gauge, generate_latest
from flask import Flask, Response, request, jsonify, redirect, send_file
from threading import Thread
from flask_cors import CORS
import subprocess
import threading
import shutil
import json
import os




# Flask Server:
app: Flask = Flask(__name__)
CORS(app)




# Global Variables:
collectorRegistry: CollectorRegistry = CollectorRegistry()

prometheus_isBenchmarkRunning: Gauge = Gauge("prometheus_isBenchmarkRunning", "Is Benchmark Running", registry=collectorRegistry)
prometheus_isBenchmarkRunning.set(0)

prometheus_benchmarkProgressBar: Gauge = Gauge("prometheus_progressBar", "Completion Bar for Last Benchmark", registry=collectorRegistry)
prometheus_benchmarkProgressBar.set(0)

setup_numberOfWorkers_value: str = "3"
setup_dataset_value: str = "IOT_DNL"
setup_mlFramework_value: str = "tf"
setup_flFramework_value: str = "ds"
setup_batchSize_value: str = "1024"
setup_patience_value: str = "5"
setup_targetScore_value: str = "None"
setup_model_value: str = "IOT_DNL"
setup_optimizer_value: str = "adam"
setup_learningRate_value: str = "0.001"
setup_comm_value: str = "mpi"
setup_loss_value: str = "scc"
setup_mainMetric_value: str = "None"
setup_seed_value: str = "42"
setup_delta_value: str = "0.01"
setup_epochs_value: str = "10"
setup_localEpochs_value: str = "3"
setup_verbosity_value: str = "Normal_Verbosity"
setup_oversubscribe_value: str = "false"
setup_hyperthreading_value: str = "false"

prometheus_currentBenchmarkMetrics_time: Gauge = Gauge("prometheus_currentBenchmarkMetrics_time", "Current Benchmark Metrics: time", ['epoch'], registry=collectorRegistry)
prometheus_currentBenchmarkMetrics_mainMetric: Gauge = Gauge("prometheus_currentBenchmarkMetrics_mainMetric", "Current Benchmark Metrics: mainMetric (MCC/SMAPE)", ['epoch'], registry=collectorRegistry)
prometheus_currentBenchmarkMetrics_as: Gauge = Gauge("prometheus_currentBenchmarkMetrics_as", "Current Benchmark Metrics: as", ['epoch'], registry=collectorRegistry)
prometheus_currentBenchmarkMetrics_f1s: Gauge = Gauge("prometheus_currentBenchmarkMetrics_f1s", "Current Benchmark Metrics: f1s", ['epoch'], registry=collectorRegistry)

benchmarkProcess = None
thread_benchmark: Thread = None
existingResultsFolderStructure: set = None
benchmarkFolderPath: str = None
benchmarkFolderPath_createdNow: bool = False
cancelButtonClicked: bool = False
lock = threading.Lock()




# Flask Methods:
@app.route('/', methods=['GET'])
def home():
	return redirect("http://localhost:3000/public-dashboards/5372a4549e8a4853a1794b0a2aaf3503")

@app.route('/startBenchmark', methods=['GET'])
def startBenchmark():
	projectRoot: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

	if ([dir for dir in os.listdir(os.path.join(projectRoot, "Data")) if os.path.isdir(os.path.join(os.path.join(projectRoot, "Data"), dir))] == []):
		os.chdir(projectRoot)

		prometheus_isBenchmarkRunning.set(1)
		prometheus_benchmarkProgressBar.set(0)

		def getDatasets():
			command = f"python3 Programs/preprocess.py -d all"
			subprocess.run(command, shell=True, executable='/usr/bin/bash')
		thread_getDatasets: Thread = Thread(target=getDatasets)
		thread_getDatasets.start()
		thread_getDatasets.join()

		def divideDatasets():
			command = f"python3 Programs/division.py -d all -n 0"
			subprocess.run(command, shell=True, executable='/usr/bin/bash')
		thread_divideDatasets: Thread = Thread(target=divideDatasets)
		thread_divideDatasets.start()
		thread_divideDatasets.join()

		prometheus_isBenchmarkRunning.set(0)

	def executeProgram() -> None:
		os.chdir(projectRoot)

		flArgs = f"-d {setup_dataset_value} -m {setup_model_value} --ml {setup_mlFramework_value} -c {setup_comm_value} --fl {setup_flFramework_value} --batch_size {setup_batchSize_value} --optimizer {setup_optimizer_value} --patience {setup_patience_value} --seed {setup_seed_value} --loss {setup_loss_value} --delta {setup_delta_value} --local_epochs {setup_localEpochs_value} --learning_rate {setup_learningRate_value} --epochs {setup_epochs_value}"
		if (setup_targetScore_value != "None"): flArgs += f" --target_score {setup_targetScore_value}"
		if (setup_mainMetric_value != "None"): flArgs += f" --main_metric {setup_mainMetric_value}"
		if (setup_verbosity_value == "Full_Verbosity"): flArgs += f" --verbose"
		
		mpirunArgs = f"-n {setup_numberOfWorkers_value}"
		if (setup_oversubscribe_value == "true"): mpirunArgs += f" --oversubscribe"
		if (setup_hyperthreading_value == "true"): mpirunArgs += f" --use-hwthread-cpus"

		command = f"source venv/bin/activate && mpirun {mpirunArgs} python3 Programs/fl.py {flArgs}"
		
		prometheus_benchmarkProgressBar.set(0)
		prometheus_isBenchmarkRunning.set(1)
		global benchmarkProcess
		benchmarkProcess = subprocess.Popen(command, shell=True, executable='/usr/bin/bash')
		benchmarkProcess.wait()
		prometheus_isBenchmarkRunning.set(0)

		global cancelButtonClicked
		if (cancelButtonClicked == False):
			global benchmarkFolderPath
			if (benchmarkFolderPath != None):
				for root, dirs, files in os.walk(benchmarkFolderPath + '_backup', topdown=False):
					for name in files:
						if (os.path.exists(os.path.join(root, name))):
							os.remove(os.path.join(root, name))
					for name in dirs:
						if (os.path.exists(os.path.join(root, name))):
							os.rmdir(os.path.join(root, name))
				shutil.rmtree(benchmarkFolderPath + '_backup')
				benchmarkFolderPath = None

		global benchmarkFolderPath_createdNow
		benchmarkFolderPath_createdNow = False

	global thread_benchmark
	thread_benchmark = Thread(target=executeProgram)
	thread_benchmark.start()

	return '', 200

@app.route('/cancelBenchmark', methods=['GET'])
def cancelBenchmark():
	global cancelButtonClicked
	cancelButtonClicked = True

	if (thread_benchmark != None and thread_benchmark.is_alive()): benchmarkProcess.kill()

	global benchmarkFolderPath
	if (benchmarkFolderPath != None and os.path.exists(benchmarkFolderPath) and os.path.exists(benchmarkFolderPath + '_backup')):
		for root, dirs, files in os.walk(benchmarkFolderPath, topdown=False):
			for name in files:
				if (os.path.exists(os.path.join(root, name))):
					os.remove(os.path.join(root, name))
			for name in dirs:
				if (os.path.exists(os.path.join(root, name))):
					os.rmdir(os.path.join(root, name))
		shutil.rmtree(benchmarkFolderPath)

		shutil.move(benchmarkFolderPath + '_backup', benchmarkFolderPath)
		benchmarkFolderPath = None

	return '', 200

@app.route('/benchmarkProgressBar', methods=['POST'])
def benchmarkProgressBar():
	data: dict = request.get_json()
	prometheus_benchmarkProgressBar.set(
		((data['benchmarkProgressBar'] / data['benchmarkProgressBarMaxValue']) * 100) if data['benchmarkStatus'] == True else 100
	)
	return '', 200

@app.route('/benchmarkMetrics', methods=['POST'])
def benchmarkMetrics():
	data: dict = request.get_json()
	prometheus_currentBenchmarkMetrics_time.labels(epoch=data['epoch']).set(data['time'])
	prometheus_currentBenchmarkMetrics_mainMetric.labels(epoch=data['epoch']).set(data['mainMetric']['MCC'] if 'MCC' in data['mainMetric'] else data['mainMetric']['SMAPE'])
	prometheus_currentBenchmarkMetrics_as.labels(epoch=data['epoch']).set(data['as'])
	prometheus_currentBenchmarkMetrics_f1s.labels(epoch=data['epoch']).set(data['f1s'])
	return '', 200

@app.route('/setup_numberOfWorkers', methods=['POST'])
def setup_numberOfWorkers():
	global setup_numberOfWorkers_value
	setup_numberOfWorkers_value = request.get_data(as_text=True).replace("workers_", "")
	return '', 200

@app.route('/setup_dataset', methods=['POST'])
def setup_dataset():
	global setup_dataset_value
	setup_dataset_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_mlFramework', methods=['POST'])
def setup_mlFramework():
	global setup_mlFramework_value
	setup_mlFramework_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_flFramework', methods=['POST'])
def setup_flFramework():
	global setup_flFramework_value
	setup_flFramework_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_batchSize', methods=['POST'])
def setup_batchSize():
	global setup_batchSize_value
	setup_batchSize_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_patience', methods=['POST'])
def setup_patience():
	global setup_patience_value
	setup_patience_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_targetScore', methods=['POST'])
def setup_targetScore():
	global setup_targetScore_value
	setup_targetScore_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_model', methods=['POST'])
def setup_model():
	global setup_model_value
	setup_model_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_optimizer', methods=['POST'])
def setup_optimizer():
	global setup_optimizer_value
	setup_optimizer_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_learningRate', methods=['POST'])
def setup_learningRate():
	global setup_learningRate_value
	setup_learningRate_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_comm', methods=['POST'])
def setup_comm():
	global setup_comm_value
	setup_comm_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_loss', methods=['POST'])
def setup_loss():
	global setup_loss_value
	setup_loss_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_mainMetric', methods=['POST'])
def setup_mainMetric():
	global setup_mainMetric_value
	setup_mainMetric_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_seed', methods=['POST'])
def setup_seed():
	global setup_seed_value
	setup_seed_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_delta', methods=['POST'])
def setup_delta():
	global setup_delta_value
	setup_delta_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_epochs', methods=['POST'])
def setup_epochs():
	global setup_epochs_value
	setup_epochs_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_localEpochs', methods=['POST'])
def setup_localEpochs():
	global setup_localEpochs_value
	setup_localEpochs_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_verbosity', methods=['POST'])
def setup_verbosity():
	global setup_verbosity_value
	setup_verbosity_value = request.get_data(as_text=True)
	return '', 200

@app.route('/setup_oversubscribe', methods=['GET'])
def setup_oversubscribe():
	global setup_oversubscribe_value
	setup_oversubscribe_value = request.args.get('status', type=str)
	return '', 200

@app.route('/setup_hyperthreading', methods=['GET'])
def setup_hyperthreading():
	global setup_hyperthreading_value
	setup_hyperthreading_value = request.args.get('status', type=str)
	return '', 200

@app.route('/getKerasFiles', methods=['GET'])
def file_structure():
	files: list = []
	if (not os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', 'Results'))): return '', 200
	directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Results'))
	def get_structure(directory):
		structure = {}
		for entry in os.listdir(directory):
			if ('_backup' in entry): continue
			path = os.path.join(directory, entry)
			if os.path.isdir(path):
				structure[entry] = get_structure(path)
			else:
				structure[entry] = path
				files.append(path)
		return structure
	get_structure(directory)

	returnList: list = []
	for eachFile in files:
		if (".keras" in eachFile):
			returnList.append(eachFile.replace(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Results')), ""))

	return jsonify(returnList)

@app.route('/downloadFile', methods=['GET'])
def download():
	filePath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Results')) + request.args.get('file')
	if not filePath or not os.path.isfile(filePath):
		return "No such file!", 400
	return send_file(filePath, as_attachment=True)

@app.route('/viewMasterLog', methods=['GET'])
def viewMasterLog():
	filePath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Results')) + request.args.get('file')
	if not filePath or not os.path.isfile(filePath):
		return "No such file!", 400

	response: dict = {}
	with open(filePath, 'r') as file:
		while (True):
			lineAsString: str = file.readline()
			if (lineAsString == ""): break

			lineAsDict: dict = json.loads(lineAsString)
			if (lineAsDict['event'] == 'results'):
				response[lineAsDict['epoch']] = {}
				response[lineAsDict['epoch']]['timestamp'] = lineAsDict['timestamp']
				response[lineAsDict['epoch']]['time'] = lineAsDict['time']
				if ('MCC' in lineAsDict): response[lineAsDict['epoch']]['MCC'] = lineAsDict['MCC']
				elif ('SMAPE' in lineAsDict): response[lineAsDict['epoch']]['SMAPE'] = lineAsDict['SMAPE']
				response[lineAsDict['epoch']]['AS'] = lineAsDict['AS']
				response[lineAsDict['epoch']]['F1S'] = lineAsDict['F1S']
	
	return jsonify(response)

@app.route('/makeFolderBackup', methods=['GET'])
def makeFolderBackup():
	with lock:
		path: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', request.args.get('path', type=str)))
		global benchmarkFolderPath_createdNow
		if (benchmarkFolderPath_createdNow == False and os.path.exists(path)):
			global benchmarkFolderPath
			benchmarkFolderPath = path
			if (not os.path.exists(benchmarkFolderPath + '_backup')):
				shutil.copytree(benchmarkFolderPath, benchmarkFolderPath + '_backup')
		else:
			benchmarkFolderPath_createdNow = True
	return '', 200

@app.route('/saveSetupConfigurations', methods=['POST'])
def save_setup_configurations():
	data = request.get_json()
	if not data:
		return jsonify({"error": "Invalid data"}), 400
	
	with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmarkSetupConfiguration.json")), "w") as f:
		json.dump(data, f, indent=4)

	return jsonify({"message": "Configuration saved successfully"}), 200

@app.route('/getSetupConfigurations', methods=['GET'])
def get_setup_configurations():
	if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmarkSetupConfiguration.json"))):
		return jsonify({"message": "No saved configuration"}), 200

	with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmarkSetupConfiguration.json")), "r") as f:
		setup_config = json.load(f)

	return jsonify(setup_config), 200




# Prometheus Scraping Method:
@app.route('/metrics')
def metrics() -> None:
	return Response(generate_latest(collectorRegistry), mimetype="text/plain")




# Support Functions:
def load_setup_configurations():

	# se não existir não faz load -> mantem valores default
	if not os.path.exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmarkSetupConfiguration.json"))):
		return

	global setup_numberOfWorkers_value
	global setup_dataset_value
	global setup_mlFramework_value
	global setup_flFramework_value
	global setup_batchSize_value
	global setup_patience_value
	global setup_targetScore_value
	global setup_model_value
	global setup_optimizer_value
	global setup_learningRate_value
	global setup_comm_value
	global setup_loss_value
	global setup_mainMetric_value
	global setup_seed_value
	global setup_delta_value
	global setup_epochs_value
	global setup_localEpochs_value
	global setup_verbosity_value
	global setup_oversubscribe_value
	global setup_hyperthreading_value

	with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "benchmarkSetupConfiguration.json")), "r") as f:
		setup_config = json.load(f)

	setup_numberOfWorkers_value = setup_config.get("numberOfWorkers", setup_numberOfWorkers_value)
	setup_dataset_value = setup_config.get("dataset", setup_dataset_value)
	setup_mlFramework_value = setup_config.get("mlFramework", setup_mlFramework_value)
	setup_flFramework_value = setup_config.get("flFramework", setup_flFramework_value)
	setup_batchSize_value = setup_config.get("batchSize", setup_batchSize_value)
	setup_patience_value = setup_config.get("patience", setup_patience_value)
	setup_targetScore_value = setup_config.get("targetScore", setup_targetScore_value)
	setup_model_value = setup_config.get("model", setup_model_value)
	setup_optimizer_value = setup_config.get("optimizer", setup_optimizer_value)
	setup_learningRate_value = setup_config.get("learningRate", setup_learningRate_value)
	setup_comm_value = setup_config.get("comm", setup_comm_value)
	setup_loss_value = setup_config.get("loss", setup_loss_value)
	setup_mainMetric_value = setup_config.get("mainMetric", setup_mainMetric_value)
	setup_seed_value = setup_config.get("seed", setup_seed_value)
	setup_delta_value = setup_config.get("delta", setup_delta_value)
	setup_epochs_value = setup_config.get("epochs", setup_epochs_value)
	setup_localEpochs_value = setup_config.get("localEpochs", setup_localEpochs_value)
	setup_verbosity_value = setup_config.get("verbosity", setup_verbosity_value)
	setup_oversubscribe_value = setup_config.get("oversubscribe", setup_oversubscribe_value)
	setup_hyperthreading_value = setup_config.get("hyperthreading", setup_hyperthreading_value)




# Program Init:
if __name__ == '__main__':
	load_setup_configurations()
	app.run(debug=True)