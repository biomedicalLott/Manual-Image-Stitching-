# Manual-Image-Stitching-
A little project to stitch images together using SIFT features from OpenCV.  
Challenge: Stitch images together using SIFT features manually. 

This will stitch together all files that share the same filename. If you were to call stitch('t2',N = 3, savepath='stitched_t2.png') it would probably put it together. It will look for all files in the path with the name "name_number", find the best order to stitch them in, and then stitch them together. 

e.g. 
<p><img src="https://i.imgur.com/W5REWoN.png"/></p>
<p><img src="https://i.imgur.com/PKdIY50.png"/></p>

becomes 
<p><img src="https://i.imgur.com/Mucxr2u.png"/></p>

Feathering would be smart to include there, but as of now it is functional! 
