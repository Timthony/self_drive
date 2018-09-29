import os
import io
import glob
import time
import threading
import picamera
import picamera.array
from PIL import Image
import numpy as np

import car_control
from keras.models import load_model
import tensorflow as tf
	
def get_max_prob_num(predictions_array):
	"""to get the integer of predition, instead of digit number"""
	
	prediction_edit = np.zeros([1,5])
	for i in range(0,5):
		if predictions_array[0][i] == predictions_array.max():
			prediction_edit[0][i] = 1
			return i
	return 2

def control_car(action_num):
	"""out put the char and call car_control(we used before)"""

	if action_num == 0:
		print("Left")
		car_control.car_turn_left()
		time.sleep(0.25)
	elif action_num== 1:
		print("Right")
		car_control.car_turn_right()
		time.sleep(0.25)
	elif action_num == 2:
		car_control.car_move_forward()
		print('Forward')
	elif action_num == 3:
		car_control.car_move_backward()
		print('Backward')
	else:
		car_control.car_stop()
		print('Stop')


class ImageProcessor(threading.Thread):
	def __init__(self, owner):
		super(ImageProcessor, self).__init__()
		self.stream = io.BytesIO()
		self.event = threading.Event()
		self.terminated = False
		self.owner = owner
		self.start()

	def run(self):
		global latest_time, model, graph
		# This method runs in a separate thread
		while not self.terminated:
			# Wait for an image to be written to the stream
			if self.event.wait(1):
				try:
					self.stream.seek(0)
					# Read the image and do some processing on it
					image = Image.open(self.stream)
					image_np = np.array(image)
					camera_data_array = np.expand_dims(image_np,axis = 0)
					current_time = time.time()
					if current_time>latest_time:
						if current_time-latest_time>1:
							print("*" * 30)
							print(current_time-latest_time)
							print("*" * 30)
						latest_time = current_time
						with graph.as_default():
							predictions_array = model.predict(camera_data_array, batch_size=20, verbose=1)
						print(predictions_array)
						action_num = get_max_prob_num(predictions_array)
						control_car(action_num)
						# Uncomment this line if you want to save images with prediction as name
						# Warning: This will cause latency sometimes.
						# image.save('%s_image%s.jpg' % (action_num,time.time()))
				finally:
					# Reset the stream and event
					self.stream.seek(0)
					self.stream.truncate()
					self.event.clear()
					# Return ourselves to the available pool
					with self.owner.lock:
						self.owner.pool.append(self)

class ProcessOutput(object):
	def __init__(self):
		self.done = False
		# Construct a pool of 4 image processors along with a lock
		# to control access between threads
		self.lock = threading.Lock()
		self.pool = [ImageProcessor(self) for i in range(4)]
		self.processor = None

	def write(self, buf):
		if buf.startswith(b'\xff\xd8'):
			# New frame; set the current processor going and grab
			# a spare one
			if self.processor:
				self.processor.event.set()
			with self.lock:
				if self.pool:
					self.processor = self.pool.pop()
				else:
					# No processor's available, we'll have to skip
					# this frame; you may want to print a warning
					# here to see whether you hit this case
					self.processor = None
		if self.processor:
			self.processor.stream.write(buf)

	def flush(self):
		# When told to flush (this indicates end of recording), shut
		# down in an orderly fashion. First, add the current processor
		# back to the pool
		if self.processor:
			with self.lock:
				self.pool.append(self.processor)
				self.processor = None
		# Now, empty the pool, joining each thread as we go
		while True:
			with self.lock:
				try:
					proc = self.pool.pop()
				except IndexError:
					pass # pool is empty
			proc.terminated = True
			proc.join()


def main():
	"""get data, then predict the data, edited data, then control the car"""
	global model, graph
	
	model_loaded = glob.glob('model/*.h5')
	for single_mod in model_loaded:
		model = load_model(single_mod)
	graph = tf.get_default_graph()
	
	try:
		with picamera.PiCamera(resolution=(160,120)) as camera:
			# uncomment this line and the camera images will be upside down
			# camera.vflip = True
			time.sleep(2)
			output = ProcessOutput()
			camera.start_recording(output, format='mjpeg')
			while not output.done:
				camera.wait_recording(1)
			camera.stop_recording()
	finally:
		car_control.cleanGPIO()

if __name__ == '__main__':
	global latest_time
	latest_time = time.time()
	main()
