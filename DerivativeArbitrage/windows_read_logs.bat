taskkill /F /IM excel.exe
python .\deribit_portoflio.py
start excel ".\Runtime\runs\deribit.xlsx


::python .\side_scripts.py ::"log_reader"
::start excel ".\Runtime\logs\latest_exec.xlsx"