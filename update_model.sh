#! /bin/bash
mkdir -p model
if [ "$(ls -A model)" ]; then
	echo "Model exists, skip downloading!!"
else
	echo "Downloading model..."
	wget -O model.zip https://vgm.cloud/s/oAzS4HcLrLFRnx7/download/translate-model.zip
	unzip model.zip -d "$(pwd)"
	rm model.zip
fi
