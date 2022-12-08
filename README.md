# SIFT--
> A very hacky approximation of SIFT.

## usage

This code depends on `scikit-image`, `matplotlib`, `opencv`, and `numpy`. To run SIFT (with the default parameters) on the two example images given in the repo, run 
```
python3 sift.py
```
It'll display both images with the keypoints overlayed - two points in different images having the same color means that they were identified as a match by the program (spoiler: it's usually wrong!)

If you've already run the program before, you can run
```
python3 sift.py --fromcache
```
to skip the extrema-finding step (which is the bottleneck)

## writeup

TL;DR
- Started by implementing the scale space / difference functions while guessing constants
- Extrema finding function couldn't find any extrema! 
  - Tried preprocessing with contrast filter
  - Turns out the scale space blurring parameters are very important!
- Realized that the chance of me finishing on time was low, so I started approximating steps of SIFT.
- Instead of a gradient histogram, just averaged out the gradients near a keypoint to get reference orientation
- Instead of a proper descriptor, just take the averaging gradients in regions nearby a point
- Instead of principal curvature, use a Hough transform anr remove points that are too close to lines
- Still doesn't work very well! Next steps:
  - Proper matching across image scales
  - Adjusting the descriptor's direction to be in terms of the reference orientation
	

Parts of this will be a terser summary of my conceptual writeup for the object detection assignment. Sorry!

Entering this, I knew how SIFT identified and described keypoints at a high level without any technical details. What I was still fuzzy on was how specifically some of these things were done, and how the matching process worked (and generally how the gap between SIFT output and an actionable homography was crossed). 

I started out by implementing what I knew: first, some preprocessing with a greyscale and a 2x upscale using bilinear interpolation. We needed to generate a scale space, which would mean repeatedly blurring the image using a Gaussian kernel, downscaling one of them, and then doing it all over again until the images got to small. This generates *octaves*: the rows of increasingly blurred images at different fixed sizes. The broad goal with this is to achieve scale invariance. Originally, I guessed constants.

The next step was taking the differences between images in each octave so that we could approximate the Laplacian. This part was simple.

The harder part was finding the extrema: points in these difference images that were the max out of pixels nearby them both in their image and the images next to them in the octave. Naive implementation was simple but it resulted in no extrema for the test image I was using. Maddeningly, it worked for obvious test cases with clear extrema, and so it didn't seem to be any sort of logic error with the extrema finding. I started tweaking everything about the scale space function and preprocessing I could find. 

I experimented with the kernel size and standard deviation of the Gaussian blur. I added a contrast filter before everything to hopefully accentuate the extremas. Combined, these measures made a few extrema be detected, but not nearly as many as the sample implementations of SIFT had. Digging through the source for one of the websites I'd originally used for my object detection writeup, I found that choosing a value for the Gaussian blur standard deviation was more complicated than I thought: there were some specialized functions just for generating it from some magic numbers. Even incorporating that, I didn't get nearly enough extrema. It was only after looking at another sample SIFT implementation that I noticed they seperated the kernel into X and Y filters, which was the jump needed to make the number of extrema reasonable. My guess is that all these hoops surrounding the exact way in which you blur things are probably consequences of the specific derivations motivating SIFT. 

Finally, with the ability to move onto the next parts of SIFT, I realized that there was little chance I was going to finish implementing the entirety of the algorithm. So, I started cutting wherever I could.

Traditionally, there's a refinement step where a Taylor approximation of the scale space function is used to get more precise points than discrete pixel positions we already had. The math was somewhat confusing, and since all it does is improve the accuracy, I sought to skip it.

The SIFT reference orientation is found by building a histogram of the gradients of pixels near the point. The goal here is to eventually build a position/rotation invariant representation of the point's surroundings. I approximated this part by skipping the histogram and just averaging the nearby gradients (which should result in a similar enough answer to still serve the same purpose as the original technique).

This reference orientation is then used to build up a descriptor, which does a lot of the same logic as the reference orientation step, but in regions farther from the point (and in the reference frame of the reference orientation). I chose to just collect the gradient at specific shifts away from the point in the interest of time. The more problematic part of this approximation is that the gradient directions aren't relative to the reference orientation like the intended step!

Now, we have keypoints with proper descriptors, and we have to match them. The full SIFT matching process involves a lot of error correction, but the basic step for checking if two keypoints are match is if the Euclidean distance between their descriptors is less than a given threshold. 

Simply picking a reasonably low threshold results in a *lot* of matches, the majority of which are just incorrect:

![img](https://i.imgur.com/aWEaqFI.png)

Part of this is because we skipped an important step: SIFT usually filters out keypoints that lie along lines/edges because they're invariant to translation - they're also especially bad for our more gradient-centric method since the gradient direction for edges are going to be similar to other edges in most cases. SIFT removes these points via *principal curvatures*, aka confusing math I don't get, but I figured I could get similar results by using the Hough transform on to find lines. 

So, I ran the image through a Canny edge detector (so that the Hough results would be less noisy), then ran a Hough transform to get the lines in the image. From there, I found the minimum distance to a line segment for all the keypoints, and filtered out any that were too close. One downside of this method is that without a bunch of parameter tuning, this step can be overeager (removing actually good keypoints because it thinks there's a line nearby).

Now we have a lot less keypoints (a *lot* less... that Hough transform needs tuning) and less of these problematic edge keypoints (we still have some, likely due to where the Hough transform decides the line is):

![img2](https://i.imgur.com/OsauNf5.png)


From here, it's a matter of trying to figure out what additions are needed to make the keypoint matches better. I tried including a Euclidean distance factor for the reference orientations at one point, but that didn't help much (and intuitively so, as that should vary with the image's rotation). I'm currently throwing away keypoints for smaller octaves in the scale space, and I could instead do the keypoint matching across each one (ultimately multiplying the position of it by some power of 2 to get the actual location). The number of found keypoints for other octaves is small enough that I figured it not worth the effort given the time I have. 

The most immediate step is probably to make it so the descriptor is actually rotation invariant, as that's what's probably killing the algorithm. One good way to test this would be to use two images of the same orientation (but different scales/lighting) and see if it suddenly works way better...

If I had much more time, I'd properly implement SIFT in its entirity instead of using these hacky approximations.
