#!/bin/bash
whisperx ${audio_file} \
    --model large-v2 \
    --language en \
    --output_dir ${output_dir} \
    --output_format scp
