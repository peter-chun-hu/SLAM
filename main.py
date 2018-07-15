from scipy import io
import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
import cv2

def get_rgb(file_name):
	key_names_rgb = ['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','image']
	# image size: 1080x1920x3 uint8
	data = io.loadmat(file_name+".mat")
	data = data['RGB'][0]
	rgb = []
	for m in data:
		tmp = {v:m[0][0][i] for (i,v) in enumerate(key_names_rgb)}
		rgb.append(tmp)
	return rgb

def replay_rgb(rgb_data):
	for k in range(len(rgb_data)):
		R = rgb_data[k]['image']
		R = np.flip(R,1)
		plt.imshow(R)		
		plt.draw()
		plt.pause(0.001)


def get_depth(file_name):
	key_names_depth = ['t','width','imu_rpy','id','odom','head_angles','c','sz','vel','rsz','body_height','tr','bpp','name','height','depth']
	data = io.loadmat(file_name+".mat")
	data = data['DEPTH'][0]
	depth = []
	for m in data:
		tmp = {v:m[0][0][i] for (i,v) in enumerate(key_names_depth)}
		depth.append(tmp)
	return depth

def replay_depth(depth_data):
	DEPTH_MAX = 4500
	DEPTH_MIN = 400	
	for k in range(len(depth_data)):
		D = depth_data[k]['depth']
		D = np.flip(D,1)
		for r in range(len(D)):
			for (c,v) in enumerate(D[r]):
				if (v<=DEPTH_MIN) or (v>=DEPTH_MAX):
					D[r][c] = 0.0
		plt.imshow(D)		
		plt.draw()
		plt.pause(0.001)


def getExtrinsics_IR_RGB():
	# The following define a transformation from the IR to the RGB frame: 
	rgb_R_ir = np.array( [
		[0.99996855100876,0.00589981445095168,0.00529992291318184],
		[-0.00589406393353581,0.999982024861347,-0.00109998388535087],
		[-0.00530631734715523,0.00106871120747419,0.999985350318977]])  
	rgb_T_ir = np.array([0.0522682,0.0015192,-0.0006059]) # meters
	return {'rgb_R_ir':rgb_R_ir, 'rgb_T_ir':rgb_T_ir}


def getIRCalib():
	'''For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/'''
	#-- Focal length:
	fc = np.array([364.457362485643273,364.542810626989194])
	#-- Principal point:
	cc = np.array([258.422487561914693,202.487139940005989])
	#-- Skew coefficient:
	alpha_c = 0.000000000000000
	#-- Distortion coefficients:
	kc = np.array([0.098069182739161,-0.249308515140031,0.000500420465085,0.000529487524259,0.000000000000000])
	#-- Focal length uncertainty:
	fc_error = np.array([1.569282671152671 , 1.461154863082004 ])
	#-- Principal point uncertainty:
	cc_error = np.array([2.286222691982841 , 1.902443125481905 ])
	#-- Skew coefficient uncertainty:
	alpha_c_error = 0.000000000000000
	#-- Distortion coefficients uncertainty:
	kc_error = np.array([0.012730833002324 , 0.038827084194026 , 0.001933599829770 , 0.002380503971426 , 0.000000000000000 ])
	#-- Image size: nx x ny
	nxy = np.array([512,424])
	return {'fc':fc, 'cc':cc, 'ac':alpha_c, 'kc':kc, 'nxy':nxy,
			'fce':fc_error, 'cce':cc_error, 'ace':alpha_c_error, 'kce':kc_error}


def getRGBCalib():
	'''For more information regarding the calibration model visit http://www.vision.caltech.edu/bouguetj/calib_doc/'''
	#-- Focal length:
	fc = np.array([1049.331752604831308,1051.318476285322504])
	#-- Principal point:
	cc = np.array([956.910516428015740,533.452032441484675])
	#-- Skew coefficient:
	alpha_c = 0.000000000000000
	#-- Distortion coefficients:
	kc = np.array([0.026147836868708 , -0.008281285819487 , -0.000157005204226 , 0.000147699131841 , 0.000000000000000])
	#-- Focal length uncertainty:
	fc_error = np.array([2.164397369394806 , 2.020071561303139 ])
	#-- Principal point uncertainty:
	cc_error = np.array([3.314956924207777 , 2.697606587350414 ])
	#-- Skew coefficient uncertainty:
	alpha_c_error = 0.000000000000000
	#-- Distortion coefficients uncertainty:
	kc_error = np.array([0.005403085916854 , 0.015403918092499 , 0.000950699224682 , 0.001181943171574 , 0.000000000000000 ])
	#-- Image size: nx x ny
	nxy = np.array([1920,1080])
	return {'fc':fc, 'cc':cc, 'ac':alpha_c, 'kc':kc, 'nxy':nxy,
		'fce':fc_error, 'cce':cc_error, 'ace':alpha_c_error, 'kce':kc_error}  

def bresenham2D(sx, sy, ex, ey):
	sx = int(round(sx))
	sy = int(round(sy))
	ex = int(round(ex))
	ey = int(round(ey))
	dx = abs(ex-sx)
	dy = abs(ey-sy)
	steep = abs(dy)>abs(dx)
	if steep:
		dx,dy = dy,dx # swap 

	if dy == 0:
		q = np.zeros((dx+1,1))
	else:
		q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
	if steep:
		if sy <= ey:
			y = np.arange(sy,ey+1)
		else:
			y = np.arange(sy,ey-1,-1)
		if sx <= ex:
			x = sx + np.cumsum(q)
		else:
			x = sx - np.cumsum(q)
	else:
		if sx <= ex:
			x = np.arange(sx,ex+1)
		else:
			x = np.arange(sx,ex-1,-1)
		if sy <= ey:
			y = sy + np.cumsum(q)
		else:
			y = sy - np.cumsum(q)
	return np.vstack((x,y))

def mapCorrelation(im, x_im, y_im, vp, xs, ys):
	nx = im.shape[0]
	ny = im.shape[1]
	xmin = x_im[0]
	xmax = x_im[-1]
	xresolution = (xmax-xmin)/(nx-1)
	ymin = y_im[0]
	ymax = y_im[-1]
	yresolution = (ymax-ymin)/(ny-1)
	nxs = xs.size
	nys = ys.size
	cpr = np.zeros((nxs, nys))
	for jy in range(0,nys):
		y1 = vp[1,:] + ys[jy] # 1 x 1076
		iy = np.int16(np.round((y1-ymin)/yresolution))
		for jx in range(0,nxs):
			x1 = vp[0,:] + xs[jx] # 1 x 1076
			ix = np.int16(np.round((x1-xmin)/xresolution))
			valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)), np.logical_and((ix >=0), (ix < nx)))
			cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
	return cpr

def get_lidar(file_name):
	data = io.loadmat(file_name+".mat")
	lidar = []
	for m in data['lidar'][0]:
		tmp = {}
		tmp['t']= m[0][0][0]
		nn = len(m[0][0])
		if (nn != 5) and (nn != 6):			
			raise ValueError("different length!")
		tmp['pose'] = m[0][0][nn-4]
		tmp['res'] = m[0][0][nn-3]
		tmp['rpy'] = m[0][0][nn-2]
		tmp['scan'] = m[0][0][nn-1]
		
		lidar.append(tmp)
	return lidar

def get_joint(file_name):
	key_names_joint = ['acc', 'ts', 'rpy', 'gyro', 'pos', 'ft_l', 'ft_r', 'head_angles']
	data = io.loadmat(file_name+".mat")
	joint = {kn: data[kn] for kn in key_names_joint}
	return joint

def initialize_table():
	radius_diff = 270/180*np.pi/1080
	start = -135/180 * np.pi
	sin_table = []
	cos_table = []
	for i in range(1081):
		sin_table
		cos_table.append(np.cos(start+i*radius_diff))
		sin_table.append(np.sin(start+i*radius_diff))
	sin_table = np.array(sin_table)
	cos_table = np.array(cos_table)
	return sin_table, cos_table

def initialize_beta(alpha):
	beta = []
	for i in range(len(alpha)):
		beta.append(np.log(alpha[i]))
	beta = np.array(beta)
	return beta

def initialize_map():
	MAP = {}
	MAP['res']   = 0.1 #meters
	MAP['xmin']  = -30  #meters
	MAP['ymin']  = -30
	MAP['xmax']  = 30
	MAP['ymax']  = 30
	MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
	MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
	MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']))
	return MAP

def convert_lidar(lidar_data, lidar_file, index, sin_table, cos_table):
	cartesian = []
	for j in range(len(lidar_data[index]['scan'][0])):
		r = lidar_data[index]['scan'][0][j]
		if(r < 0.1 or r > 30.0):
			continue
		x = r * cos_table[j]
		y = r * sin_table[j]
		cartesian.append([x,y])
	cartesian = np.array(cartesian)
	return cartesian

def generate_rotation_matrix(joint_data, index):
	neck_angle = joint_data['head_angles'][0][index]
	head_angle = joint_data['head_angles'][1][index]
	z_axis = np.array([[np.cos(neck_angle), -np.sin(neck_angle), 0, 0], [np.sin(neck_angle), np.cos(neck_angle), 0, 0], [0, 0 ,1, 0.48], [0, 0, 0, 1]])
	y_axis = np.array([[np.cos(head_angle), 0, np.sin(head_angle), 0], [0, 1, 0, 0], [-np.sin(head_angle), 0, np.cos(head_angle), 0], [0, 0, 0, 1]])
	rotation_matrix = y_axis.dot(z_axis)
	return rotation_matrix

def convert_body_frame(joint_data, lidar_data, cartesian_table, index, joint_index):
	body_frame = np.array([[],[],[],[]])
	last_diff = 10000000
	current_diff = abs(lidar_data[index]['t'][0][0] - joint_data['ts'][0][joint_index])
	while(current_diff < last_diff and joint_index < len(joint_data['ts'][0])):
		last_diff = current_diff
		joint_index = joint_index + 1
		current_diff = abs(lidar_data[index]['t'][0][0] - joint_data['ts'][0][joint_index])
	joint_index = joint_index - 1
	rotation_matrix = generate_rotation_matrix(joint_data, joint_index)
	for j in range(len(cartesian_table)):
		temp = np.array([[cartesian_table[j][0]], [cartesian_table[j][1]], [0], [1]])
		body_frame = np.hstack((body_frame, rotation_matrix.dot(temp)))
	return body_frame, rotation_matrix

def convert_world_frame(body_frame, X_temp):
	x = X_temp[0]
	y = X_temp[1]
	theta = X_temp[2]
	body2world = np.array([[np.cos(theta), -np.sin(theta), 0, x], [np.sin(theta), np.cos(theta), 0, y], [0, 0 ,1, 0.93], [0, 0, 0, 1]])
	world_frame = body2world.dot(body_frame)
	world_frame = world_frame[0:3, world_frame[2,:] > 0.1]
	return world_frame

def convert_cell_frame(world_frame, robot_map, X):
	minus_num = [[robot_map['xmin']], [robot_map['ymin']], [0]]
	position = np.ceil((np.transpose(X) - minus_num) / robot_map['res']) - 1
	position = position[0][0:2]
	cell_frame = np.ceil((world_frame - minus_num) / robot_map['res']) - 1
	cell_frame = cell_frame.astype(np.int16)
	return cell_frame, position

def initialize_filter():
	particle_num = 100
	mu = 0
	sigma = 0.1
	X = []
	alpha = []
	X.append([0.0,0.0,0.0])
	for i in range(particle_num-1):
		X.append([0.0,0.0,0.0])
		alpha.append(1/particle_num)
	alpha.append(1/particle_num)
	X = np.array(X)
	alpha = np.array(alpha)
	return X, alpha, particle_num

def filter_predict(X, lidar_data, particle_num, index, rotation_matrix, O_last):
	position_mu = 0
	position_sigma = 0.01
	angle_mu = 0
	angle_sigma = 1/180*np.pi

	x = lidar_data[index]['pose'][0][0]
	y = lidar_data[index]['pose'][0][1]
	theta = lidar_data[index]['pose'][0][2]
	wOl = np.array([[np.cos(theta), -np.sin(theta), 0, x], [np.sin(theta), np.cos(theta), 0, y], [0, 0 ,1, 0.48], [0, 0, 0, 1]])
	O = wOl.dot(np.linalg.inv(rotation_matrix))
	x_diff = O[0][3] - O_last[0][3]
	y_diff = O[1][3] - O_last[1][3]
	theta_diff = mat2euler(O[0:3][0:3])[2] - mat2euler(O_last[0:3][0:3])[2]
	O_diff = np.tile([x_diff, y_diff, theta_diff],(particle_num,1))

	w = np.random.normal(position_mu, position_sigma, (particle_num, 2))
	angle_random = np.random.normal(angle_mu, angle_sigma, (particle_num, 1))
	X[:,0:2] = X[:,0:2] + O_diff[:,0:2] + w[:,0:2]
	X[:,2] = X[:,2] + O_diff[:,2] + angle_random[:,0]

	return O, theta_diff

def filter_update(X, beta, robot_map, body_frame, recovered_map, x_im, y_im, x_range, y_range, theta_diff_sum):
	accuracy = 1
	# theta_num = max(np.ceil(theta_diff_sum / 20 / (accuracy*np.pi / 180)).astype(np.int16) * 2 + 1, 1)
	theta_num = 5
	temp_degree = accuracy*np.pi / 180
	for i in range(len(beta)):
		X_temp = np.tile(X[i][:], (theta_num,1))
		theta_range1 = np.arange(-(theta_num - 1) / 2 * temp_degree, -0.5 * temp_degree, temp_degree)[:, np.newaxis]
		theta_range2 = np.arange(temp_degree, ((theta_num - 1) / 2 + 0.5) * temp_degree, temp_degree)[:, np.newaxis]
		theta_range = np.vstack(([[0]], theta_range1, theta_range2))
		X_temp[:, 2] = X_temp[:, 2] + theta_range[:,0]
		max_value = float("-infinity")
		for j in range(theta_num):
			world_frame = convert_world_frame(body_frame, X_temp[j])
			world_frame[2][:] = 0.0
			temp_score = mapCorrelation(recovered_map, x_im, y_im, world_frame, x_range, y_range)
			temp_max_index = np.unravel_index(temp_score.argmax(), temp_score.shape)
			if(temp_score[temp_max_index[0], temp_max_index[1]] > max_value):
				max_value = temp_score[temp_max_index[0], temp_max_index[1]]
				theta_index = j
				max_index = [temp_max_index[0], temp_max_index[1]]
				origin = temp_score[4][4]
			if(j == 0 and max_value > 300):
				break;

		X[i][:] = X_temp[theta_index,:]
		if(max_value != origin):
			X[i] = X[i] + np.array([(max_index[0] - 4)*0.1, (max_index[1] - 4)*0.1, 0])
		beta[i] = beta[i] + max_value

	max_beta_index = np.argmax(beta)
	world_frame = convert_world_frame(body_frame, X[max_beta_index])
	cell_frame, temp_position = convert_cell_frame(world_frame, robot_map, X[max_beta_index])
	max_beta = max(beta)
	constant = 0
	for i in range(len(beta)):
		constant = constant + np.exp(beta[i] - max_beta)
	constant = np.log(constant)
	for i in range(len(beta)):
		beta[i] = beta[i] - max_beta - constant
	return beta, cell_frame, X, temp_position.astype(np.int16)

def claculate_thredshold(alpha, beta):
	for i in range(len(beta)):
		alpha[i] = np.exp(beta[i])
	sigma = sum(np.power(alpha, 2))
	return 1 / sigma

def resample(X, alpha, beta, particle_num):
	c = alpha[0]
	j = 0
	new_alpha = []
	new_X = []
	for k in range(particle_num):
		u = np.random.uniform(0, 1/particle_num)
		b = u + k/particle_num
		while(b > c):
			j = j + 1
			c = c + alpha[j]
		new_X.append(X[j])
		new_alpha.append(1/particle_num)
	new_X = np.array(new_X)
	new_alpha = np.array(new_alpha)

	alpha[:] = new_alpha[:]
	X[:] = new_X[:]
	beta = np.log(alpha)
	return X, beta

def bresenham(X, beta, cell_frame, robot_map):
	max_beta_index = np.argmax(beta)
	start_x = np.ceil((X[max_beta_index][0]-robot_map['xmin']) / robot_map['res']).astype(np.int16)-1
	start_y = np.ceil((X[max_beta_index][1]-robot_map['ymin']) / robot_map['res']).astype(np.int16)-1
	empty_cell = [[],[]]
	empty_cell = np.array(empty_cell).astype(np.int16)
	for i in range(len(cell_frame[0])):
		temp_empty_cell = bresenham2D(start_x, start_y, cell_frame[0][i], cell_frame[1][i],)
		temp_empty_cell = np.delete(temp_empty_cell, -1, 1).astype(np.int16)
		empty_cell = np.hstack((empty_cell, temp_empty_cell))
	return empty_cell

def update_map(robot_map, empty_cell, cell_frame):
	b = 4
	minus = -np.log(b)
	plus = np.log(b)
	for i in range(len(empty_cell[0])):
		map_x = empty_cell[0][i]
		map_y = empty_cell[1][i]
		robot_map['map'][map_x][map_y] = robot_map['map'][map_x][map_y] + minus
	for i in range(len(cell_frame[0])):
		map_x = cell_frame[0][i]
		map_y = cell_frame[1][i]
		robot_map['map'][map_x][map_y] = robot_map['map'][map_x][map_y] + plus
	return robot_map

def recover_map(robot_map, recovered_map):
	D = 0.8
	B = 0.2
	recovered_map = np.zeros((robot_map['sizex'], robot_map['sizey']))
	temp_map = np.ones((robot_map['sizex'], robot_map['sizey']))
	robot_map['map'][robot_map['map'] > 700] = 700
	temp_map = temp_map - 1/(1+np.exp(robot_map['map']))
	recovered_map[temp_map > D] = 1
	recovered_map[temp_map < B] = -1
	return recovered_map

def find_time(lidar_time, data, index):
	last_diff = 10000000
	current_diff = abs(lidar_time - data[index]['t'])
	while(current_diff < last_diff):
		last_diff = current_diff
		index = index + 1
		current_diff = abs(lidar_time - data[index]['t'])
	index = index - 1
	return index

def test(test_map, position, k):
	plt.clf()
	new_map = np.uint8((-test_map+1)/2*255)
	new_map = cv2.cvtColor(new_map, cv2.COLOR_GRAY2RGB)
	position_array = np.array(position)
	a = np.shape(position)
	for i in range(a[0]):
		new_map[position_array[i][0], position_array[i][1], 0] = 255
		new_map[position_array[i][0], position_array[i][1], 1] = 0
		new_map[position_array[i][0], position_array[i][1], 2] = 0
	new_map = np.rot90(new_map)
	imgplot = plt.imshow(new_map)
	plt.imsave(str(k)+'.png', new_map)
	plt.draw()
	plt.pause(0.000001)

def	build_rgb_map(rgb_map, rgb_data, depth_data, depth_index, u, v):
	depth_data[depth_index]['t'][0][0]

	return rgb_map
	
if __name__ == "__main__":
	lidar_file = "../trainset/lidar/train_lidar0"
	joint_file = "../trainset/joint/train_joint0"
	# rgb_file = "../trainset/cam/RGB_1"
	# depth_file = "../trainset/cam/DEPTH_1"
	
	# exIR_RGB = getExtrinsics_IR_RGB()
	# IRCalib = getIRCalib()
	# RGBCalib = getRGBCalib()

	lidar_data = get_lidar(lidar_file)
	joint_data = get_joint(joint_file)
	# rgb_data = get_rgb(rgb_file)
	# depth_data = get_depth(depth_file)
    # u = np.tile(np.arange(-255.5, 256, 1),(424,1))
    # v = np.tile(np.arange(-211.5, 212, 1),(1, 512))

	sin_table, cos_table =  initialize_table()
	X, alpha, particle_num = initialize_filter()
	joint_index = 0
	O_last = np.identity(4)
	beta = initialize_beta(alpha)
	robot_map = initialize_map()
	recovered_map = np.zeros((robot_map['sizex'], robot_map['sizey']))
	rgb_map = np.zeros((robot_map['sizex'], robot_map['sizey'], 3))
	x_im = np.arange(robot_map['xmin'],robot_map['xmax']+robot_map['res']/2,robot_map['res']) #x-positions of each pixel of the map
	y_im = np.arange(robot_map['ymin'],robot_map['ymax']+robot_map['res']/2,robot_map['res']) #y-positions of each pixel of the map
	x_range = np.arange(-0.4,0.4+0.1/2,0.1)
	y_range = np.arange(-0.4,0.4+0.1/2,0.1)
	rgb_index = 0
	depth_index = 0
	position = []
	theta_diff_sum = 0
	for i in range(0, len(lidar_data), 10):
		print(i/len(lidar_data)*100)
		cartesian = convert_lidar(lidar_data, lidar_file, i, sin_table, cos_table)
		body_frame, rotation_matrix = convert_body_frame(joint_data, lidar_data, cartesian, i, joint_index)
		O_last, theta_diff = filter_predict(X, lidar_data, particle_num, i, rotation_matrix, O_last)
		theta_diff_sum = theta_diff_sum + abs(theta_diff)
		if(i % 100 == 0):
			beta, cell_frame, X, temp_position = filter_update(X, beta, robot_map, body_frame, recovered_map, x_im, y_im, x_range, y_range, theta_diff_sum)
			theta_diff_sum = 0
			position.append([temp_position[0], temp_position[1]])
			if(claculate_thredshold(alpha, beta) < particle_num / 20):
				X, beta = resample(X, alpha, beta, particle_num)
			empty_cell = bresenham(X, beta, cell_frame, robot_map)
			robot_map = update_map(robot_map, empty_cell, cell_frame)
			recovered_map = recover_map(robot_map, recovered_map)
		# depth_index = find_time(lidar_data[i]['t'][0][0], depth_data, depth_index)
		# if(abs(depth_data[depth_index]['t'][0][0] - lidar_data[i]['t'][0][0]) < 1):
			# rgb_map = build_rgb_map(rgb_map, rgb_data, depth_data, depth_index, u, v)
		if(i % 1000 == 0 and i != 0):
			test(recovered_map, position, i)
	test(recovered_map, position, 0)
	input()