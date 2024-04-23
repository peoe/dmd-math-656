default: data run

data:
	@echo "Fetching data..."
	curl -o ./pedestrian.zip http://jacarini.dinf.usherbrooke.ca/static/pedestrian%20detection/pedestrian%20detection%20dataset.zip && unzip ./pedestrian.zip -d data && rm pedestrian.zip
	curl -o ./canoe.zip http://jacarini.dinf.usherbrooke.ca/static/dataset/dynamicBackground/canoe.zip && unzip ./canoe.zip -d data && rm canoe.zip

all: data
	@echo "Running scripts to generate GIFs, plots, and matrices..."
	
	@echo "Running pedestrian motion detection..."
	@python pedestrian_motion_detection.py --output=all
	
	@echo "Running pedestrian foreground detection..."
	@python pedestrian_fg_detection.py --output=all
	
	@echo "Running canoe motion detection..."
	@python canoe_motion_detection.py --output=all
	
	@echo "Running canoe foreground detection..."
	@python canoe_fg_detection.py --output=all
	
	@echo "Running sofa motion detection..."
	@python sofa_motion_detection.py --output=all
	
	@echo "Running sofa foreground detection..."
	@python sofa_fg_detection.py --output=all

gifs: data
	@echo "Running scripts to generate GIFs..."

	@echo "Running pedestrian motion detection..."
	@python pedestrian_motion_detection.py --output=gifs
	
	@echo "Running pedestrian foreground detection..."
	@python pedestrian_fg_detection.py --output=gifs
	
	@echo "Running canoe motion detection..."
	@python canoe_motion_detection.py --output=gifs
	
	@echo "Running canoe foreground detection..."
	@python canoe_fg_detection.py --output=gifs
	
	@echo "Running sofa motion detection..."
	@python sofa_motion_detection.py --output=gifs
	
	@echo "Running sofa foreground detection..."
	@python sofa_fg_detection.py --output=gifs

plots: data
	@echo "Running scripts to generate plots..."
	
	@echo "Running pedestrian motion detection..."
	@python pedestrian_motion_detection.py --output=plots
	
	@echo "Running pedestrian foreground detection..."
	@python pedestrian_fg_detection.py --output=plots
	
	@echo "Running canoe motion detection..."
	@python canoe_motion_detection.py --output=plots
	
	@echo "Running canoe foreground detection..."
	@python canoe_fg_detection.py --output=plots
	
	@echo "Running sofa motion detection..."
	@python sofa_motion_detection.py --output=plots
	
	@echo "Running sofa foreground detection..."
	@python sofa_fg_detection.py --output=plots

matrices: data
	@echo "Running scripts to generate matrices..."
	
	@echo "Running pedestrian motion detection..."
	@python pedestrian_motion_detection.py --output=matrices
	
	@echo "Running pedestrian foreground detection..."
	@python pedestrian_fg_detection.py --output=matrices
	
	@echo "Running canoe motion detection..."
	@python canoe_motion_detection.py --output=matrices
	
	@echo "Running canoe foreground detection..."
	@python canoe_fg_detection.py --output=matrices
	
	@echo "Running sofa motion detection..."
	@python sofa_motion_detection.py --output=matrices
	
	@echo "Running sofa foreground detection..."
	@python sofa_fg_detection.py --output=matrices
