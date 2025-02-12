import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
CHECKERBOARD = (10, 7)
SQUARE_SIZE = 16.0  # mm
BALL_DIAMETER = 40.0  # mm, MUST BE ACCURATE!
CALIBRATION_DATA_PATH = "calibration_data.npz"

class CameraCalibration:
    """Stores camera calibration data."""
    def __init__(self, mtx, dist):
        self.mtx = mtx
        self.dist = dist

def draw_axes(img, rvec, tvec, camera_calibration, length):
    """Draws camera axes on the image."""
    imgpts, _ = cv2.projectPoints(np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,-length]]), rvec, tvec, camera_calibration.mtx, camera_calibration.dist)
    imgpts = imgpts.astype(int)
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 5)  # X-axis (Red)
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 5)  # Y-axis (Green)
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 5)  # Z-axis (Blue)
    return img

def init_plot(checkerboard_size, square_size):
    """Initializes the Matplotlib plot."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Checkerboard Points ---
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    ax.checkerboard_plot = ax.scatter(objp[:, 0], objp[:, 1], objp[:, 2], c='b', marker='o', label='Checkerboard')

    # --- Camera Position and Axes ---
    ax.camera_position_plot = ax.scatter([], [], [], c='r', marker='x', s=100, label='Camera')
    ax.camera_x_axis_plot = ax.quiver([], [], [], [], [], [], color='r', length=square_size * checkerboard_size[0] / 2, label='Camera X')
    ax.camera_y_axis_plot = ax.quiver([], [], [], [], [], [], color='g', length=square_size * checkerboard_size[0] / 2, label='Camera Y')
    ax.camera_z_axis_plot = ax.quiver([], [], [], [], [], [], color='b', length=square_size * checkerboard_size[0] / 2, label='Camera Z')

    # --- Ball Position (initialize) ---
    # We will add ball positions dynamically, so we don't initialize a scatter plot here

    # --- Labels, Title, Legend, View ---
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Camera, Checkerboard, and Ball Pose')
    ax.legend()


    # --- Set axis limits and aspect ratio for a better view ---
    ax.set_xlim([0, square_size * checkerboard_size[0]])
    ax.set_ylim([0, square_size * checkerboard_size[1]])
    ax.set_zlim([-square_size * checkerboard_size[0]/2, square_size * checkerboard_size[0]/2]) # Adjust as needed
    # THIS IS CRUCIAL:  Set the box aspect ratio to be proportional to the world units
    ax.set_box_aspect([checkerboard_size[0] * square_size, checkerboard_size[1] * square_size, checkerboard_size[0] * square_size])

    # Initial camera view (elevation and azimuth angles)
    ax.view_init(elev=20, azim=-135)  # Good starting point

    return fig, ax

def update_camera_pose(ax, rvec, tvec):
    """Updates only the camera pose in the plot."""
    if rvec is not None and tvec is not None:
        R, _ = cv2.Rodrigues(rvec)
        camera_position = -np.matrix(R).T @ tvec
        camera_position = camera_position.flatten()
        camera_x_axis = R[:, 0]
        camera_y_axis = R[:, 1]
        camera_z_axis = R[:, 2]

        ax.camera_position_plot._offsets3d = ([camera_position[0,0]], [camera_position[0,1]], [camera_position[0,2]])
        ax.camera_x_axis_plot.set_segments([[[camera_position[0,0], camera_position[0,1], camera_position[0,2]],
                                                [camera_position[0,0] + camera_x_axis[0],
                                                camera_position[0,1] + camera_x_axis[1],
                                                camera_position[0,2] + camera_x_axis[2]]]])
        ax.camera_y_axis_plot.set_segments([[[camera_position[0,0], camera_position[0,1], camera_position[0,2]],
                                                [camera_position[0,0] + camera_y_axis[0],
                                                camera_position[0,1] + camera_y_axis[1],
                                                camera_position[0,2] + camera_y_axis[2]]]])
        ax.camera_z_axis_plot.set_segments([[[camera_position[0,0], camera_position[0,1], camera_position[0,2]],
                                                [camera_position[0,0] + camera_z_axis[0],
                                                camera_position[0,1] + camera_z_axis[1],
                                                camera_position[0,2] + camera_z_axis[2]]]])


def reconstruct_3d_single_camera(u, v, camera_calibration, rvec, tvec, ball_diameter):
    """Calculates 3D position from a single camera using known object size."""
    # 1. Back-projection (ray casting)
    p_img = np.array([u, v, 1]).reshape(3, 1)
    p_cam = np.linalg.inv(camera_calibration.mtx) @ p_img

    # 2.  We do not know the scale, but we can estimate using the ball information.

    # 3. 3D point in camera coordinates (before scaling)
    # P_camera = s * p_cam  # We need to find 's'

    # 4. 3D Point in World Coordinates, using the extrinsic parameters.
    R, _ = cv2.Rodrigues(rvec)
    # P_world = R @ P_camera + t
    # P_world = R @ (s * p_cam) + t
    # P_world = s * (R @ p_cam) + t
    P_world_unscaled = R @ p_cam
    t = tvec.reshape(3,1)

    # 5. Find S
    # We know that P_world[2] / (P_world_unscaled[2] * s + t[2])  = 0
    # 0 = (P_world_unscaled[2] * s + t[2])
    # -t[2] = P_world_unscaled[2] * s
    s = -t[2,0] / P_world_unscaled[2,0]

    P_world = s * P_world_unscaled + t

    return P_world.flatten()


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to YOLO model file', required=True)
    parser.add_argument('--source', help='Image source', required=True)
    parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5, type=float)
    parser.add_argument('--resolution', help='Resolution in WxH', default=None)
    parser.add_argument('--record', help='Record results', action='store_true')
    parser.add_argument('--iou', help='IOU threshold for tracking', default=0.3, type=float)
    parser.add_argument('--tracker', help='Tracker type (botsort or bytetrack)', default="bytetrack.yaml", type=str)
    args = parser.parse_args()

    # --- Parse Inputs ---
    model_path = args.model
    img_source = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record = args.record
    iou_thresh = args.iou
    tracker_type = args.tracker

    # --- Model Loading and Validation ---
    if not os.path.exists(model_path):
        print('ERROR: Model path is invalid.')
        sys.exit(1)

    model = YOLO(model_path, task='detect')
    labels = model.names

    # --- Source Type Determination ---
    img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
    vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if ext in img_ext_list:
            source_type = 'image'
        elif ext in vid_ext_list:
            source_type = 'video'
        else:
            print(f'File extension {ext} is not supported.')
            sys.exit(1)
    elif 'usb' in img_source:
        source_type = 'usb'
        usb_idx = int(img_source[3:])
    elif 'picamera' in img_source:
        source_type = 'picamera'
        picam_idx = int(img_source[8:])
    else:
        print(f'Input {img_source} is invalid.')
        sys.exit(1)

    # --- Resolution Parsing ---
    resize = False
    if user_res:
        resize = True
        try:
            resW, resH = map(int, user_res.split('x'))
        except ValueError:
            print("Invalid resolution format.  Use WxH (e.g., 640x480)")
            sys.exit(1)

    # --- Recording Setup ---
    if record:
        if source_type not in ['video', 'usb']:
            print('Recording only works for video and camera sources.')
            sys.exit(1)
        if not user_res:
            print('Please specify resolution to record video at.')
            sys.exit(1)
        record_name = 'demo1.avi'
        record_fps = 30
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

     # --- Source Loading ---
    if source_type == 'image':
        imgs_list = [img_source]
    elif source_type == 'folder':
        imgs_list = glob.glob(os.path.join(img_source, '*'))
        imgs_list = [f for f in imgs_list if os.path.splitext(f)[1] in img_ext_list]
    elif source_type == 'video' or source_type == 'usb':
        cap_arg = img_source if source_type == 'video' else usb_idx
        cap = cv2.VideoCapture(cap_arg)
        if user_res:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    elif source_type == 'picamera':
        from picamera2 import Picamera2
        cap = Picamera2()
        cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
        cap.start()

    # --- Bounding Box Colors ---
    bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
                   (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)]

    # --- Control Variables ---
    avg_frame_rate = 0
    frame_rate_buffer = []
    fps_avg_len = 200
    img_count = 0
    frame_number = 0

    # --- Load Calibration Data (using np.load) ---
    try:
        calibration_data = np.load(CALIBRATION_DATA_PATH)
        mtx = calibration_data['mtx']
        dist = calibration_data['dist']
        camera_calibration = CameraCalibration(mtx, dist)
        print("Loaded Camera Matrix:\n", mtx)
        print("Loaded Distortion Coefficients:\n", dist)

    except FileNotFoundError:
        print(f"Error: Calibration data file not found at {CALIBRATION_DATA_PATH}.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading calibration data: {e}")
        sys.exit(1)

    # --- Prepare Object Points (3D) for Checkerboard ---
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # --- Initialize Matplotlib Plot ---
    fig, ax = init_plot(CHECKERBOARD, SQUARE_SIZE)

    plt.show(block=False)

    # --- Store Ball Positions and Debugging Counter ---
    ball_positions = []
    plot_count = 0


    # --- Main Loop ---
    while True:
        t_start = time.perf_counter()

        # --- Frame Loading ---
        if source_type == 'image' or source_type == 'folder':
            if img_count >= len(imgs_list):
                print('All images processed.')
                break
            img_filename = imgs_list[img_count]
            frame = cv2.imread(img_filename)
            img_count += 1
        elif source_type in ('video', 'usb'):
            ret, frame = cap.read()
            if not ret:
                print('End of video or camera error.')
                break
        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            if frame is None:
                print('Picamera error.')
                break
        frame_number += 1

        if resize:
            frame = cv2.resize(frame, (resW, resH))

        # --- Undistort the Frame ---
        undistorted_frame = cv2.undistort(frame, camera_calibration.mtx, camera_calibration.dist, None, camera_calibration.mtx)

        # --- Chessboard Detection and Pose Estimation ---
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        ret_cb, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        rvec = None  # Initialize rvec and tvec
        tvec = None
        if ret_cb:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            ret_solvepnp, rvec, tvec, _ = cv2.solvePnPRansac(objp, corners2, camera_calibration.mtx, camera_calibration.dist)

            if ret_solvepnp:
                frame = draw_axes(undistorted_frame, rvec, tvec, camera_calibration, SQUARE_SIZE * 3)
                # Update camera pose in the plot
                update_camera_pose(ax, rvec, tvec)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        # --- Tracking Inference (on Undistorted Frame) ---
        results = model.track(undistorted_frame, persist=True, verbose=False, tracker=tracker_type, iou=iou_thresh)

        detections = results[0].boxes
        object_count = 0
        ball_3d_position = None # Initialize

        if results[0].boxes.id is not None:
            for i in range(len(detections)):
                if detections[i].id is not None:
                    xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
                    xmin, ymin, xmax, ymax = xyxy

                    classidx = int(detections[i].cls.item())
                    classname = labels[classidx]
                    conf = detections[i].conf.item()
                    track_id = int(detections[i].id.item())

                    if conf > min_thresh:
                        color = bbox_colors[classidx % 10]
                        cv2.rectangle(undistorted_frame, (xmin, ymin), (xmax, ymax), color, 2)

                        label = f'{classname} ID:{track_id} {int(conf * 100)}%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_ymin = max(ymin, labelSize[1] + 10)
                        cv2.rectangle(undistorted_frame, (xmin, label_ymin - labelSize[1] - 10),
                                      (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                        cv2.putText(undistorted_frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        object_count += 1

                        # --- 3D Position Estimation (Object) ---
                        center_x = int((xmin + xmax) / 2)
                        center_y = int((ymin + ymax) / 2)
                        cv2.circle(undistorted_frame, (center_x, center_y), 5, (0, 255, 0), -1)

                        if ret_cb and ret_solvepnp:
                            ball_3d_position = reconstruct_3d_single_camera(center_x, center_y, camera_calibration, rvec, tvec, BALL_DIAMETER)

        # --- Display and Frame Rate ---
        if source_type in ('video', 'usb', 'picamera'):
            cv2.putText(undistorted_frame, f'FPS: {avg_frame_rate:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        cv2.putText(undistorted_frame, f'Number of objects: {object_count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
        cv2.imshow('YOLO Tracking', undistorted_frame)

        # --- Key Handling ---
        key = cv2.waitKey(1)

        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            cv2.waitKey()
        elif key in (ord('p'), ord('P')):
            cv2.imwrite('capture.png', undistorted_frame)
        elif key == ord('t') and ball_3d_position is not None:
            ball_positions.append(ball_3d_position)  # Store the position
            plot_count += 1
            print(f"Plot {plot_count}: Ball position = {ball_3d_position}") # Debugging output

            # Update the 3D plot with all stored ball positions
            if ball_positions:  # Check if the list is not empty
                ball_positions_array = np.array(ball_positions)
                ax.scatter(ball_positions_array[:, 0], ball_positions_array[:, 1], ball_positions_array[:, 2], c='m', marker='o', s=50)
                fig.canvas.draw_idle()  # Redraw the canvas
                fig.canvas.flush_events()

        # --- FPS Calculation ---
        t_stop = time.perf_counter()
        frame_rate_calc = 1 / (t_stop - t_start)
        if len(frame_rate_buffer) >= fps_avg_len:
            frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
        avg_frame_rate = np.mean(frame_rate_buffer)

    # --- Cleanup ---
    print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
    plt.show(block=True)  # Keep the plot window open *and* block
    if source_type in ('video', 'usb'):
        cap.release()
    elif source_type == 'picamera':
        cap.stop()
    if record:
        recorder.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()