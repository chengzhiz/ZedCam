This is the project for automatic CV data logging using Zed Camera in the museum setting. 

# Hardware equipments:
* ZED 2i Stereo Camera
https://www.stereolabs.com/store/products/zed-2i
* ZED Box
https://www.stereolabs.com/store/products/zed-box-orin
Total: $519 + $ 1299 
* Others: 
keyboard and mouse, portable monitor, tripod for mounting the Zed Camera and get it in place

# Code explaination:
###  V3: 
tracking human and lingering time in defined space by drawing the area before recording. Export the following when quitting the program. 
* CSV with the ID, Entry and Exit time, Lingering duration, Space (A/B/C), 
* Accelerated recorded video
### V4: 
Automatically checkpoint saved the video per 30 minutes, progressively save the CSV file per 20 minutes.
### V5: 
Automatically saved the checkpoint video and progressively save the csv file every 20 minutes. 
* Now it can record the lingering time accurately
* UI showing the next checkpoint time, current time
* "q" for quitting the program


