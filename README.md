This is the project for automatic data logging using Zed Camera in the museum 

# Hardware equipments:
* ZED 2i Stereo Camera
https://www.stereolabs.com/store/products/zed-2i
* ZED Box
https://www.stereolabs.com/store/products/zed-box-orin
Total: $519 + $ 1299 
* Others: 
keyboard and mouse, portable monitor, camera stand for Zed Camera

# Code explaination:
###  V3: tracking human and lingering time in defined space by drawing the area before recording. Export the following when quitting the program. 
* CSV with the ID, Entry and Exit time, Lingering duration, Space (A/B/C), 
* accelerated recorded video
### V4: Automatically checkpoint saved the video per 30 minutes, progressively save the CSV file per 30 minutes
### V5 (To-do)
* Debug the lingering time record inaccuracy
* Better UI
* "q" for quitting the program


