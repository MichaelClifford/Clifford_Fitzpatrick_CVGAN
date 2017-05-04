import numpy as np
import random
import linecache
import time
import imageio # for saving gifs


from keras.models import Sequential
from keras.layers import Conv3D, Conv2D, Conv2DTranspose, Activation, Dense, Merge, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import tanh
from keras.layers.normalization import BatchNormalization
from keras_contrib.layers import Deconvolution3D
from keras.layers.core import Flatten
from keras.backend import expand_dims, tile, ones_like, zeros_like
from keras.regularizers import l1
from keras.layers.convolutional import ZeroPadding3D, ZeroPadding2D
import tensorflow as tf


class ElapsedTimer(object):
	'''
	This implements a timer to keep track of training time.
	'''
	def __init__(self):
		self.start_time = time.time()
	def elapsed(self,sec):
		if sec < 60:
			return str(sec) + " sec"
		elif sec < (60 * 60):
			return str(sec / 60) + " min"
		else:
			return str(sec / (60 * 60)) + " hr"
	def elapsed_time(self):
		print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class VGAN(object):
	'''
	This class implements the generator, discriminator, and the GAN networks, independent of dataset.
	'''
	def __init__(self, vid_rows=64, vid_cols=64, num_frames=32, channels=3):
		self.vid_rows = vid_rows
		self.vid_cols = vid_cols
		self.num_frames = num_frames
		self.channels = channels
		self.D = None	# discriminator
		self.G = None	# generator
		self.AM = None	# adversarial model
		self.DM = None	# discriminiator model
		self.input_shape = (self.vid_rows, self.vid_cols, self.num_frames, self.channels)

	def discriminator(self):
		'''
		This is the discriminator. 
		It's a 5 layer 3dconvolutional binary classifier 
		with batch normalization and leaky relu activations.
		'''
		if self.D:
			return self.D

		self.D = Sequential()

		# First layer
		self.D.add(Conv3D(filters=64, 
			kernel_size=(4,4,4), 
			strides=(1,1,1), 
			padding="same", 
			input_shape=self.input_shape))
		self.D.add(LeakyReLU(alpha=0.2))
		
		# Second layer
		self.D.add(Conv3D(filters=128, 
			kernel_size=(4,4,4), 
			strides=(2,2,2), 
			padding="same"))
		self.D.add(BatchNormalization())
		self.D.add(LeakyReLU(alpha=0.2))
		
		# Third layer
		self.D.add(Conv3D(filters=256, 
			kernel_size=(4,4,4), 
			strides=(2,2,2), 
			padding="same"))
		self.D.add(BatchNormalization())
		self.D.add(LeakyReLU(alpha=0.2))
		
		# Forth layer
		self.D.add(Conv3D(filters=512, 
			kernel_size=(4,4,4), 
			strides=(2,2,2), 
			padding="same"))
		self.D.add(BatchNormalization())
		self.D.add(LeakyReLU(alpha=0.2))

		# Fifth layer
		self.D.add(Conv3D(filters=2, 
			kernel_size=(4,4,4), 
			strides=(2,2,2), 
			padding="same"))
		self.D.add(BatchNormalization())
		self.D.add(LeakyReLU(alpha=0.2))

		self.D.add(Conv3D(filters=1, 
			kernel_size=(4,4,4), 
			strides=(4,4,4), 
			padding="same"))

		self.D.add(Flatten())

		# # Fifth layer
		# self.D.add(Conv3D(filters=2, 
		# 	kernel_size=(3,8,8), 
		# 	use_bias=True))
		# self.D.add(Dense(1, activation='sigmoid'))
		# self.D.add(Flatten())
		
		print("Discriminator")
		print(self.D.summary())
		print('expecting input: {} \n\n'.format(self.D.input))
		return self.D


	def generator(self):
		'''
		This is the constructor of the generator. It combines the _video() stream
		and the _static() streams with the _mask(). It uses the helper function _gen_net()
		because the foreground (video) stream forks before completion.
		'''
		if self.G:
			return self.G
			
		self.G = Sequential()
		foreground = self._video()
		# background = self._static()
		# gen_net = self._gen_net(foreground)
		# m_times_f, m_times_b = self._combine(foreground, background, gen_net)
		# self.G.add(Merge([m_times_f, m_times_b], mode='sum'))
		self.G.add(foreground)

		print("Generator")
		print(self.G.summary())
		print('expecting input: {} \n\n'.format(self.G.input))
		return self.G


	def _static(self):
		static = Sequential()

		# First layer
		static.add(Conv2DTranspose(filters=512,
							kernel_size=(4,4),
							strides = (4,4),
							padding = 'same',
							input_shape = (1,1,100))) # need to make sure this is right
		static.add(BatchNormalization())
		static.add(LeakyReLU(alpha=0.2))

		# Second layer
		static.add(Conv2DTranspose(filters=256,
							kernel_size=(4,4),
							strides = (2,2),
							padding = 'same'))
		static.add(BatchNormalization())
		static.add(LeakyReLU(alpha=0.2))

		# Third layer
		static.add(Conv2DTranspose(filters=128,
							kernel_size=(4,4),
							strides = (2,2),
							padding = 'same'))
		static.add(BatchNormalization())
		static.add(LeakyReLU(alpha=0.2))

		# Forth layer
		static.add(Conv2DTranspose(filters=64,
							kernel_size=(4,4),
							strides = (2,2),
							padding = 'same'))
		static.add(BatchNormalization())
		static.add(LeakyReLU(alpha=0.2))

		# Fifth layer
		static.add(Conv2DTranspose(filters=3,
							kernel_size=(4,4),
							strides = (2,2),
							padding = 'same',
							activity_regularizer=Activation('tanh')))
		#static.add(BatchNormalization())


		# # Sixth layer
		# static.add(Conv2DTranspose(filters=3,
		#					 kernel_size=(4,4),
		#					 strides = (2,2),
		#					 padding = 'same',))
		# static.add(BatchNormalization())
		# static.add(LeakyReLU(alpha=0.2))

		print("BACKGROUND STREAM")
		print(static.summary())
		print('expecting input: {} \n\n'.format(static.input))

		return static

	def _video(self): # make the generator the only stream. 
		'''
		The foreground stream. It learns the dynamics in the foreground.
		'''
		video = Sequential()
		video.add(Deconvolution3D(filters=512,output_shape =(None,4,4,2,512),
							kernel_size=(4,4,2),
							strides = (2,2,2),
							padding = 'valid',
							input_shape = (1,1,1,100)))
		video.add(BatchNormalization())
		video.add(LeakyReLU(alpha=0.2))

		video.add(Deconvolution3D(filters=256,output_shape =(None,8,8,4,256),
							kernel_size=(4,4,2),
							strides = (2,2,2),
							padding = 'same'))
		video.add(BatchNormalization())
		video.add(LeakyReLU(alpha=0.2))

		video.add(Deconvolution3D(filters=128,output_shape =(None,16,16,8,128),
							kernel_size=(4,4,2),
							strides = (2,2,2),
							padding = 'same'))
		video.add(BatchNormalization())
		video.add(LeakyReLU(alpha=0.2))



		video.add(Deconvolution3D(filters=64,output_shape =(None,32,32,16,64),
							kernel_size=(4,4,2),
							strides = (2,2,2),
							padding = 'same'))
		video.add(BatchNormalization())
		video.add(LeakyReLU(alpha=0.2))


		video.add(Deconvolution3D(filters=3, 
							output_shape=(None,64,64,32,3),#(None,64,64,32,3), 
							kernel_size=(4,4,4), 
							strides=(2,2,2), 
							padding='same', 
							activity_regularizer=Activation('tanh')))

		print("FOREGROUND STREAM")
		print(video.summary())
		print("expecting input: {}\n\n".format(video.input))
		return video

	# def _gen_net(self,video):
	# 	gen_net = Sequential()
	# 	gen_net.add(video)
	# 	gen_net.add(Deconvolution3D(filters=3, 
	# 	output_shape=(None,64,64,32,3),#(None,64,64,32,3), 
	# 	kernel_size=(4,4,4), 
	# 	strides=(2,2,2), 
	# 	padding='same', 
	# 	activity_regularizer=Activation('tanh')))
	# 	return gen_net

	# def _mask(self,video):
	# 	mask = Sequential()
	# 	mask.add(video)
	# 	mask.add(Deconvolution3D(filters=1, 
	# 		output_shape=(None,64,64,32,1),#(None,64,64,32,1), 
	# 		kernel_size=(4,4,2), 
	# 		strides=(2,2,2), 
	# 		padding = 'same', 
	# 		kernel_regularizer=l1(l=0.1), 
	# 		activity_regularizer=Activation('sigmoid')))
	# 	return mask

	def _combine(self,video, static, gen_net):
		# m.*f + (1-m).*b

		# m is mask
		# f is video (foreground)
		# b is static (background)

		#print('hey look: ', video.output)
		mask = self._mask(video)

		one_minus_m = Sequential()
		one_minus_m.add(mask)
		one_minus_m.add(Lambda(lambda x: tile(x,[1,1,1,1,3])))
		one_minus_m.add(Lambda(lambda x: 1-x))

		b = Sequential()
		b.add(static)
		b.add(Lambda(lambda x: expand_dims(x,axis=3)))
		b.add(Lambda(lambda x: tile(x,[1,1,1,32,1])))

		m_times_b = Sequential()
		m_times_b.add(Merge([one_minus_m, b], mode='mul'))

		mforf = Sequential()
		mforf.add(mask)
		mforf.add(Lambda(lambda x: tile(x,[1,1,1,1,3])))


		m_times_f = Merge([mforf, gen_net], mode='mul')

		return m_times_f, m_times_b


	def discriminator_model(self):
		if self.DM:
			return self.DM
		#optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
		self.DM = Sequential()
		self.DM.add(self.discriminator())
		self.DM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		

		return self.DM


	def adversarial_model(self):
		if self.AM:
			return self.AM
		#optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
		self.AM = Sequential()
		self.AM.add(self.generator())
		self.AM.add(self.discriminator())
		self.AM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return self.AM

class UCF_VGAN(object):
	'''
	Uses the adversarial model with the UCF dataset.
	Essentially this implements a training method based 
	on the specific dataset being used (UCF-101).
	'''
	
	def __init__(self, joblist):
		self.vid_rows = 128
		self.vid_cols = 128
		self.num_frames = 34
		self.channels = 3

		#self.x_train = None # READ DATA, see143 of ex
		self.x_train_joblist = joblist
		
		self.data = data_loader(self.x_train_joblist)

		self.x_train = []
		# with open(joblist, 'r') as file:
		# 	vid_path = file.readline()
		# 	vid = imageio.imread(vid_path).reshape(self.num_frame, self.vid_rows, self.vid_cols, self.channels)
		# 	self.x_train.append(vid)

		self.VGAN = VGAN()
		self.discriminator = self.VGAN.discriminator_model()
		self.adversarial = self.VGAN.adversarial_model()
		self.generator = self.VGAN.generator()



	def train(self, train_steps=2, batch_size=32, save_interval=0):
		print("STARTING TRAIN...")
		noise_input = None
		if save_interval>0:
			#noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
			pass

		for i in range(train_steps):
			
			# get a batch and reshape the images
			print('Fetching batch of size {}...'.format(batch_size))
			videos_train = self.data.get_batch(batch_size)

			print('\nCropping videos from tensor shape {}...'.format((self.channels, self.vid_rows, self.vid_cols, self.num_frames)))
			for i,video in enumerate(videos_train):
				#print('initial video shape is: {}'.format(video.shape))
				video = video.reshape(self.vid_cols, self.vid_rows, self.num_frames, self.channels)
				videos_train[i] = video[32:96, 32:96, :-2, :]
			videos_train=np.array(videos_train)
			print('new video shape is: {}'.format(videos_train[0].shape))

			print("\nGenerating noise inputs...")
			static_noise = np.random.rand(batch_size, 1, 1, 100)
			video_noise = np.expand_dims(static_noise, 2)
			print("static noise shape: {}".format(static_noise.shape))
			print("video noise shape: {}".format(video_noise.shape))
			print("generator expecting...")
			print(self.generator.input,"\n")
			#videos_fake = self.generator.predict([video_noise, static_noise])
			videos_fake = self.generator.predict(video_noise)

			print("noise batches generated from generator...\n")
			print("Combining with real video batch")
			x = np.concatenate((videos_train, videos_fake))
			print("Labeling for classification")
			
			# y = tf.concat((ones_like(videos_train, dtype=tf.uint8), 			# real label
			# 			   zeros_like(videos_fake, dtype=tf.uint8)), axis=0)   # fake label
			
			y = np.ones([2*batch_size, 1])
			y[batch_size:, :] = 0

			print("Training discriminator...")
			d_loss = self.discriminator.train_on_batch(x, y)


			#y = np.ones([batch_size, 1])
			#noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
			#noise = np.random.rand(batch_size, 1, 1, 1, 1, 100)
			#noise = [video_noise, static_noise]
			noise = video_noise
			y = [0,0,0,0,1,1,1,1]   # np.ones(2)
			
			#make_trainable(self.discriminator,False)

			print("Training adversarial net...")
			a_loss = self.adversarial.train_on_batch(noise, y)
			

			#log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
			#log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
			#log_mesg = "[A loss: %f, acc: %f]" % (a_loss[0], a_loss[1])
			log_mesg = a_loss
			print(log_mesg)
			if save_interval>0:
				if (i+1)%save_interval==0:
					#self.plot_images(save2file=True, samples=noise_input.shape[0],\
						#noise=noise_input, step=(i+1))
					self.generator.save_weights('checkpoint')
					print('saving generator weights')
				
			
			# out =  adversarial.predict(noise,batch_size=1)

			# num_frames = out.shape[3]

			# gif_frame_list = []
			
			# for i in range(num_frames):
			# 	frame = imageio.imwrite('frame.jpg', numpy.squeeze(out)[:,:,i,:])
			# 	gif_frame_list.append(imageio.imread('frame.jpg'))
			# imageio.mimsave('test.gif', gif_frame_list)


	def save_gifs(self, save2file=False, fake=True, samples=16, noise=None, step=0):
		'''
		The GIF-saving processes has been hardcoded in the single_stream model. 
		'''
				'''
		The GIF-saving processes has been hardcoded in the single_stream model. 
		'''
		pass

class data_loader(object):
	def __init__(self, joblist):
		
		self.joblist = joblist

		with open(self.joblist) as f:
			for i,l in enumerate(f):
				pass
			self.num_jobs = i+1
		print(str(self.num_jobs)+' videos in dataset...')
	
	def get_batch(self, batch_size, replacement=True):
		if replacement:
			idx = random.sample(range(1, self.num_jobs), self.num_jobs-1)

		vid_list = []
		# only accept videos with 34 frames
		while len(vid_list) < batch_size:
			vid_idx = random.sample(range(1,self.num_jobs), 1)
			vid_path = linecache.getline(self.joblist, vid_idx[0])[:-1] # get rid of hidden newline
			video =  imageio.imread(vid_path)
			if video.shape[0]==4352:
				vid_list.append(imageio.imread(vid_path))
		return vid_list


if __name__ == '__main__':
	
	# give the jobs list
	jobslist = "makeup.txt"
	
	# instantiate the model
	ucf_vgan = UCF_VGAN(jobslist)
	print('Model instantiated...')
	
	# initialize the timer
	timer = ElapsedTimer()
	
	# train the model
	ucf_vgan.train(train_steps=1, batch_size=8, save_interval=1)
	timer.elapsed_time()
	
	# save some results
	ucf_vgan.save_gifs()
	ucf_vgan.generator.load_weights('checkpoint')	# load the saved model
	static_noise = np.random.rand(1, 1, 1, 100)	# generate noise for sampling	
	video_noise = np.expand_dims(static_noise, 2)
	out = ucf_vgan.generator.predict(video_noise)	# sample
	print(out.shape)
	num_frames = out.shape[3]

	gif_frame_list = []				# save to gif
			
	for i in range(num_frames):
		frame = imageio.imwrite('frame.jpg', np.squeeze(out)[:,:,i,:])
		gif_frame_list.append(imageio.imread('frame.jpg'))
		imageio.mimsave('out_put_video.gif', gif_frame_list)

