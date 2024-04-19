default: data run

data:
	echo "Fetching data..."
	curl -o ./pedestrian.zip http://jacarini.dinf.usherbrooke.ca/static/pedestrian%20detection/pedestrian%20detection%20dataset.zip && unzip ./pedestrian.zip -d data && rm pedestrian.zip
	curl -o ./canoe.zip http://jacarini.dinf.usherbrooke.ca/static/dataset/dynamicBackground/canoe.zip && unzip ./canoe.zip -d data && rm canoe.zip

run: data
	echo "Running scripts to generate GIFs, plots, and matrices..."
	
	echo "Running pedestrian detection..."
	python pedestrian_detection.py --output=all

gifs: data
	echo "Running scripts to generate GIFs..."
	
	echo "Running pedestrian detection..."
	python pedestrian_detection.py --output=gifs

plots: data
	echo "Running scripts to generate plots..."
	
	echo "Running pedestrian detection..."
	python pedestrian_detection.py --output=plots

matrices: data
	echo "Running scripts to generate matrices..."
	
	echo "Running pedestrian detection..."
	python pedestrian_detection.py --output=matrices
