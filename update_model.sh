#! /bin/bash
mkdir -p model
if [ "$(ls -A model)" ]; then
	echo "Model exists, skip downloading!!"
else
	echo "Downloading model..."
	# Prompt the user for their shareid
	read -p "Enter your shareid: " shareid
	# Prompt the user for their password (input will be hidden)
	read -sp "Enter your password: " password
	if [ ! -z "${shareid}" ] && [ ! -z "${password}" ]; then
		wget -O model.zip --user="${shareid}" --password="${password}" --header="X-Requested-With: XMLHttpRequest" "https://vgm.cloud/public.php/webdav/"
		unzip model.zip -d "$(pwd)"
		rm model.zip
	else
		echo "shareid or password is empty"
	fi
fi
