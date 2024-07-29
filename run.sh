#!/bin/bash

# Initialize flags
api_flag=false
stt_flag=false
tts_flag=false
soni_flag=false

# Parse command-line arguments
for arg in "$@"; do
	case $arg in
	--api)
		api_flag=true
		shift
		;;
	--stt)
		stt_flag=true
		shift
		;;
	--tts)
		tts_flag=true
		shift
		;;
	--soni)
		tts_flag=true
		shift
		;;
	*)
		# Unknown option
		echo "Unknown option: $arg"
		exit 1
		;;
	esac
done

# Use the flags
if $api_flag; then
	python "api.py" &
fi

if $stt_flag; then
	python "stt.py" &
fi

if $tts_flag; then
	python "tts.py" &
fi

if $soni_flag; then
	python "app.py" &
fi

wait
