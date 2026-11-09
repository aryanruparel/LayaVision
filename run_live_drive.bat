@echo off
echo ========================================================================
echo  Starting LayaVision Live Video Autonomous Driving Perception
echo ========================================================================
echo Controls:
echo   [SPACE] = Pause / Resume
echo   [R]     = Restart Video
echo   [Q]     = Quit Playback
echo ========================================================================
.\.venv\Scripts\python.exe run_video_drive.py
pause
