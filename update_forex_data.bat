@echo off
cd /d %~dp0
python update_data.py
echo Update completed at %date% %time% >> update_log.txt 