import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
from pprint import pprint
import random
import pickle
from skimage.transform import probabilistic_hough_line
from skimage import feature, data

sigma_min = 0.8
sigma_in = 0.5
delta_min = 0.5

def gaussian(mat, sigma):
    size = math.floor(sigma * 4)
    if size % 2 == 0:
        size -= 1 # HACK
    return cv2.GaussianBlur(mat, (0, 0), sigma)


def scale_space(img, scales=3):
    width, height = img.shape[:2]
    octaves = math.floor(math.log2(min(width, height))) - 2;
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(grey)
    norm = cv2.normalize(eq, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    start = cv2.resize(
        norm,
        (height*2, width*2),
        interpolation = cv2.INTER_LINEAR
    )
    scale_space = []
    print(octaves, scales)
    for i in range(octaves):
        if i == 0:
            scale_space.append([gaussian(start, (sigma_min**2 - sigma_in**2)/delta_min)])
        else:
            scale_space.append([cv2.resize(scale_space[i-1][scales], (height, width))])
        for s in range(1,scales+3):            
            scale_space[i].append(gaussian(scale_space[i][s-1],
                math.sqrt(2**(2*s/scales)-2**(2*(s-1)/scales)) * sigma_min/delta_min
            ))
        width//=2
        height//=2

    print(len(scale_space[0]))
    return scale_space


def diff_gaussians(scale_space):
    rows = []
    for i in range(len(scale_space)):
        row = []
        for j in range(0, len(scale_space[i])-1):
            row.append(scale_space[i][j+1]-scale_space[i][j])
        rows.append(row)
    return rows


def find_extrema(gaussians):
    extremas = [[[] for _ in range(len(gaussians[0])-2)] for _ in range(len(gaussians))]
    for i in range(len(gaussians)):
        for j in range(1, len(gaussians[i])-1):
            w, h = gaussians[i][j].shape
            for k in range(1,w-1):
                for l in range(1,h-1):
                    if abs(gaussians[i][j][k][l]) < 0.01:
                        continue
                    is_max = True
                    is_min = True
                    for m in range(-1, 2):
                        for n in range(-1, 2):
                            for o in range(-1, 2):
                                if m == 0 and n == 0 and o == 0:
                                    continue
                                if gaussians[i][j+m][k+n][l+o] >= gaussians[i][j][k][l]:
                                    is_max = False
                                if gaussians[i][j+m][k+n][l+o] <= gaussians[i][j][k][l]:
                                    is_min = False                                
                    if is_max or is_min:
                        extremas[i][j-1].append((k, l))
    return extremas

def get_grad(img, x, y):
    # print(img[x][y])
    return np.array([
        img[x+1][y] - img[x][y],
        img[x][y+1] - img[x][y],
    ])

def reference_orientation(img, pt, sz=4):
    x, y = pt
    vec = np.array([0.0,0.0])
    for i in range(-sz, sz+1):
        for j in range(-sz, sz+1):
            # print(get_grad(img, x+i, y+j))
            vec += get_grad(img, x+i, y+j)
    return vec/np.linalg.norm(vec)

def descriptor(img, pt, sz=4, shift=5):
    x, y = pt
    d = np.array([0.0, 0.0]*4)    
    d[:2] = reference_orientation(img, (x+shift, y+shift))
    d[2:4] = reference_orientation(img, (x-shift, y+shift))
    d[4:6] = reference_orientation(img, (x+shift, y-shift))
    d[6:] = reference_orientation(img, (x-shift, y-shift))
    return d
        
    # vec = np.array([0.0,0.0])
    # for i in range(-sz, sz+1):
    #     for j in range(-sz, sz+1):
    #         # print(get_grad(img, x+i, y+j))
    #         vec += get_grad(img, x+i, y+j)
    # return vec/np.linalg.norm(vec)

def show_extrema(dog, i, j, extrema):
    dog = dog.copy()
    for e in extrema:
        a, b = e
        dog[i][j][a][b] = 1
    im = plt.imshow(dog[i][j], cmap="gray")
    plt.colorbar(im)
    plt.show()


def show_img(img):
    im = plt.imshow(img, cmap="gray")
    plt.colorbar(im)
    plt.show()


def preview_octaves(l):
    for i, row in enumerate(l):
        for j, img in enumerate(row):
            try:
                print(i, j)
                plt.imshow(img, cmap="gray")
                plt.show()
            except:
                pass

def run_siftmm(f):
    img = cv2.imread(f)
    width, height = img.shape[:2]
    img = cv2.resize(img, (height//8, width//8))
    space = scale_space(img)
    # preview_octaves(scale_space)

    dog = diff_gaussians(space)

    sums = [0, 0]
    for i in range(128):
        for j in range(128):
            sums[0] += dog[0][0][i][j]
            sums[1] += dog[0][1][i][j]
    print(sums)

    extremas = find_extrema(dog)
    print(len(space[0]), len(dog[0]), len(extremas[0]))
    for i in extremas:
        for j in i:
            print(len(j))


    info = []
    for i in range(3):
        important_pts = extremas[0][i]
        for pt in important_pts:
            try:
                orientation = reference_orientation(space[0][i+1], pt)
                dcptor = descriptor(space[0][i+1], pt)
                info.append((pt, orientation, dcptor))
            except IndexError:
                pass        
        
    # important_pts = extremas[0][0] + extremas[0][1] + extremas[0][2]
    # orientations = [reference_orientation(img, pt) for pt in important_pts]
    return info


def prep(f):
    img = cv2.imread(f)
    width, height = img.shape[:2]
    img = cv2.resize(img, (height//4, width//4))    
    return img

def lineseg_dists(p, a, b):
    # print(p.shape)
    # print(a.shape, b.shape)
    # """Cartesian distance from point to line segment

    # Edited to support arguments as series, from:
    # https://stackoverflow.com/a/54442561/11208892

    # Args:
    #     - p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
    #     - a: np.array of shape (x, 2)
    #     - b: np.array of shape (x, 2)
    # """
    # normalized tangent vectors
    d_ba = b - a
    d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1])
                           .reshape(-1, 1)))

    # signed parallel distance components
    # rowwise dot products of 2D vectors
    s = np.multiply(a - p, d).sum(axis=1)
    t = np.multiply(p - b, d).sum(axis=1)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, np.zeros(len(s))])

    # perpendicular distance component
    # rowwise cross products of 2D vectors  
    d_pa = p - a
    c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]

    return np.hypot(h, c)

cache = True
if cache:
    res1, res2 = pickle.load(open("./keypts.pkl", "rb"))
else:
    res1 = run_siftmm("./desk1.jpg")
    res2 = run_siftmm("./desk2.jpg")
    f = open("./keypts.pkl", "wb")
    pickle.dump((res1, res2), f)
    f.close()

def remove_lines(res, img_name, thres=20):
    
    img = prep(img_name)
    width, height = img.shape[:2]
    mask = np.zeros((width, height))
    for pt, v, d in res:
        dot_sz = 4
        for i in range(-dot_sz, dot_sz+1):
            for j in range(-dot_sz, dot_sz+1):
                mask[pt[0]+i][pt[1]+j] = 255 # (255, 255, 255)
    # show_img(np.uint8(mask))

    edge_image = feature.canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma=1, low_threshold=100, high_threshold=120)
    lines = probabilistic_hough_line(edge_image, threshold=1, line_length=40)
    # print(lines)
    # linesP = cv2.HoughLinesP(np.uint8(mask), 1, np.pi / 180, 50, None, 50, 10)
    # for i in lines:
    #     print(i)
    def show_lines(edge_image, lines):
        plt.figure(figsize = (5,5))
        plt.imshow(edge_image)
        plt.axis('off')
        for line in lines:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        plt.show()

    starts = np.array([np.array(line[0]) for line in lines])
    ends = np.array([np.array(line[1]) for line in lines])
    
    # show_lines(img, lines)
    # for x in res:
    #     print(min(lineseg_dists(np.array(x[0]), starts, ends)))
    return list(filter(lambda x: min(lineseg_dists(np.array(x[0]), starts, ends)) > thres, res))

print("BEFORE", len(res1), len(res2))
res1 = remove_lines(res1, "./desk1.jpg")
res2 = remove_lines(res2, "./desk2.jpg")
print("AFTER", len(res1), len(res2))
# img = prep("./desk2.jpg")
# edge_image = feature.canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), sigma=1, low_threshold=100, high_threshold=120)
# lines = probabilistic_hough_line(edge_image, threshold=1, line_length=20)
# starts = np.array([np.array(line[0]) for line in lines])
# ends = np.array([np.array(line[1]) for line in lines])
# res2 = list(filter(lambda x: min(lineseg_dists(np.array(x[0]), starts, ends)) < 10, res1))

# res = list(filter(lambda x: min(lineseg_dists(np.array(x[0]), starts, ends)) < 10, res1))
# for pt, v, d in res1:
#     print(min(lineseg_dists(np.array(pt), starts, ends)))
    
# print(res1, res2)

matches = []
for pt, v, d in res1:
    for pt2, v2, d2 in res2:
        v_distance = np.linalg.norm(v-v2)
        d_distance = np.linalg.norm(d-d2)
        if v_distance < 0.1 and d_distance < 0.1:
            matches.append((pt, pt2))
print(f"{len(matches)} MATCHES FOUND")
img1 = prep("./desk1.jpg")
img2 = prep("./desk2.jpg")
for pt, pt2 in matches:
    # print(pt, pt2)
    dot_sz = 4
    color = (random.random()*255, random.random()*255, random.random()*255)
    for i in range(-dot_sz, dot_sz+1):
        for j in range(-dot_sz, dot_sz+1):
            img1[pt[0]+i, pt[1]+j] = color
            img2[pt2[0]+i, pt2[1]+j] = color

f, axarr = plt.subplots(2)
axarr[0].imshow(img1)
axarr[1].imshow(img2)
plt.show()


# show_extrema(dog, 0, 1, extremas[0][0])
# show_extrema(dog, 0, 2, extremas[0][1])
# show_extrema(dog, 0, 3, extremas[0][2])

