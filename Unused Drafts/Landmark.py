# Code to access landmarks
for landmark in mp_holistic.HandLandmark:
	print(landmark, landmark.value)

print(mp_holistic.HandLandmark.WRIST.value)