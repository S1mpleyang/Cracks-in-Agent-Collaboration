python dataprocess.py --target_directory ./direct
python dataprocess.py --target_directory ./summary
python dataprocess.py --target_directory ./vote

python AS3-ACC-399-ASR.py --target_directory ./direct/process
python AS3-ACC-399-ASR.py --target_directory ./summary/process
python AS3-ACC-399-ASR.py --target_directory ./vote/process