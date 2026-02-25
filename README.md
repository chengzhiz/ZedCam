This is the project for automatic participants' interaction data logging using Zed Camera in the museum setting. 

# Hardware equipments:
* ZED 2i Stereo Camera
https://www.stereolabs.com/store/products/zed-2i
* ZED Box
https://www.stereolabs.com/store/products/zed-box-orin
Total: $519 + $ 1299 
* Others: 
keyboard and mouse, portable monitor, tripod for mounting the Zed Camera and get it in place

# Other equipments:
* No need to have Wifi connection

# Version explaination:

###  V3: 
tracking human and lingering time in defined space by drawing the area before recording. Export the following when quitting the program. 
* CSV with the ID, Entry and Exit time, Lingering duration, Space (A/B/C), 
* Accelerated recorded video
### V4: 
Automatically checkpoint saved the video per 30 minutes, progressively save the CSV file per 20 minutes.
### V6: 
Automatically saved the checkpoint video and progressively save the csv file every 20 minutes. 
* Now it can record the lingering time accurately
* UI showing the next checkpoint time, current time
* "q" for quitting the program
- sometimes it can still lose track of a person stayingg in the video record area

# Other possible improvements to be done:
* Make texts on the screen more visible 

