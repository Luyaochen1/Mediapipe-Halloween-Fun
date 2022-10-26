# References: 
# - Original model: https://google.github.io/mediapipe/solutions/face_detection.html
# - Pumpkin image: https://pixabay.com/photos/pumpkin-fruit-orange-fall-2805140/

import cv2
import mediapipe as mp
import numpy as np

import datetime

from utils.pumpkin_face_utils import read_pumpkin_image, draw_pumpkins
# from utils.fire_hair_utils import HairSegmentation, get_fire_gif
from imread_from_url import imread_from_url
from utils.face_mesh_utils import ExorcistFace
from utils.skeleton_pose_utils import SkeletonPose 

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

webcam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
webcam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow("Pumpkin face", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Pumpkin face",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

# Read pumpkin image
#pumpkin_image_path = "https://cdn.pixabay.com/photo/2017/10/01/11/36/pumpkin-2805140_960_720.png"
#pumpkin_image = read_pumpkin_image(pumpkin_image_path)

pumpkin_image_path = './images/pumpkin-2805140_960_720.png'
pumpkin_image = cv2.imread(pumpkin_image_path,cv2.IMREAD_UNCHANGED)


# Inialize background segmentation (0: small model for distace < 2m, 1: full range model for distance < 5m)
face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

#hair_segmentation = HairSegmentation(webcam_width, webcam_height)

# Read background image
#background_image_url = "https://cdn.pixabay.com/photo/2015/09/26/13/25/halloween-959047_960_720.jpg"
#background_image = imread_from_url(background_image_url)
background_image_url = './images/halloween-959047_960_720.jpg'
background_image=cv2.imread(background_image_url)

# Read smoke image
#smoke_image_url = "https://images.unsplash.com/photo-1542789828-6c82d889ed74?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&dl=eberhard-grossgasteiger-HlJZ-xm3KCI-unsplash.jpg"
#smoke_image = imread_from_url(smoke_image_url)

smoke_image_url = "./images/smoke.jpg"
smoke_image=cv2.imread(smoke_image_url)

background_image = cv2.resize(background_image, (webcam_width, webcam_height), interpolation = cv2.INTER_AREA)
smoke_image = cv2.resize(smoke_image, (webcam_width, webcam_height), interpolation = cv2.INTER_AREA)

# Inialize background segmentation (0: default model, 1: landmark image optimized)
selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(1)

show_webcam = True
max_people = 2

# Image to swap face with
# exorcist_image_url = "https://static.wikia.nocookie.net/villains/images/6/66/Theexorcisgirl.png/revision/latest?cb=20190623020548"
exorcist_image_url = "./images/Theexorcisgirl.webp"

# Initialize ExorcistFace class
draw_exorcist = ExorcistFace(exorcist_image_url, show_webcam, max_people)

# Initialize ExorcistFace class
draw_skeleton = SkeletonPose(show_webcam)


while cap.isOpened():

	# Read frame
	ret, frame = cap.read()

	img_height, img_width, _ = frame.shape

	if not ret:
		continue

	# Flip the image horizontally
	frame = cv2.flip(frame, 1)

	func = round(datetime.datetime.now().second / 12 + 0.51)
	if func == 1 : 
		# Detect face
		input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		input_image.flags.writeable = True
		detections = face_detection.process(input_image).detections

		# Draw pumkins
		output_img = draw_pumpkins(frame, pumpkin_image, detections, show_webcam)

	if func == 2 : 
		a = 1
		# Segment hair
		# hair_mask = hair_segmentation(frame)

		# Draw fire 
		# output_img = hair_segmentation.draw_fire_hair(frame, hair_mask)

	if func == 3 : 
	# Extract background
		input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		input_image.flags.writeable = False
		background_mask = cv2.cvtColor(selfie_segmentation.process(input_image).segmentation_mask,cv2.COLOR_GRAY2RGB)

		# Combine the webcam image with smoke
		smoke_frame = cv2.addWeighted(frame, 0.4, smoke_image, 0.6, 0)

		# Fill the background mask with the background image (Multiply with the mask to get a smoother combination)
		output_img = np.uint8(smoke_frame * background_mask + background_image * (1-background_mask))

	if func == 4 : 
		ret, output_img = draw_exorcist(frame)

	if func == 5 or func == 2:
 		ret, output_img = draw_skeleton(frame)

	try:
		
		cv2.imshow("Pumpkin face", output_img)
	except:
		print (func, '--error')
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

