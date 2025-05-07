import qi
import argparse
import sys
import socket
import random
import time
import numpy as np


def create_socket(args):
    s = socket.socket()
    print("Socket successfully created")
    try:
        s.bind(('', args.send_port))
        s.listen(10)
        print("Socket is listening on port: {}".format(args.send_port))
    except:
        s.close()
        print("ERR: Connection Failed, socket not bound (most likely port still open -> try killing it)")
        exit(1)
    return s


def create_speakService(session):
    return session.service("ALTextToSpeech")


def create_trackService(session):
    return session.service("ALBasicAwareness")


def create_videoService(session, args):
    video_service = session.service("ALVideoDevice")
    video_service.setParameter(args.cam_id, 35, 1)
    videoClient = video_service.subscribeCamera("Camera_{}".format(random.randint(0, 100000)),
                                                args.cam_id,
                                                args.res_id, args.color_id,
                                                args.fps)
    video_service.setResolution(str(args.cam_id), args.res_id)
    # print(help(video_service.setCamerasParameter))
    # video_service.setParameter(args.cam_id, 0, 4) # Brightness
    # video_service.setParameter(args.cam_id, 11, 0) # AutoExposure
    # video_service.setParameter(args.cam_id, 17, 1024/4) # ManualExposure

    return video_service, videoClient

def nod(session):
    print("NODDING")
    motion = session.service('ALMotion')
    current_pitch = round(motion.getAngles(['HeadPitch'], True)[0], 2)
    motion.changeAngles(['HeadPitch'], [-0.3], 0.1)
    time.sleep(1)
    motion = session.service('ALMotion')
    motion.setAngles(['HeadPitch'], [current_pitch], 0.1)

    # posture_service = session.service("ALRobotPosture")
    # posture_service.goToPosture("StandZero", 1.0)
    # posture_service.goToPosture("StandInit", 1.0)


def alive(session):
    awareness = session.service('ALBasicAwareness')
    awareness.setEnabled(False)

    background_movement = session.service('ALBackgroundMovement')
    background_movement.setEnabled(True)

    motion = session.service('ALMotion')
    motion.setStiffnesses('Head', 0.6)
    motion.setAngles(['HeadPitch', 'HeadYaw'], [0, 0], 0.1)


def idle(session):  
    # Do idle head movement with 1/3 chance
    if np.random.uniform(0, 1) >= 0.33:
        return

    motion = session.service('ALMotion')
    motion.setAngles(['HeadPitch', 'HeadYaw'],
                     [np.random.normal(0,-0.03),
                      np.random.normal(0,-0.03)],
                     0.05)


def freeze(session):
    awareness = session.service('ALBasicAwareness')
    awareness.setEnabled(False)

    background_movement = session.service('ALBackgroundMovement')
    background_movement.setEnabled(False)

    motion = session.service('ALMotion')
    motion.setStiffnesses('Head', 0.6)
    motion.setAngles(['HeadPitch', 'HeadYaw'], [-0.2, 0], 0.1)


def move_head(session, pitch, yaw):
    motion = session.service('ALMotion')
    motion.changeAngles(['HeadPitch', 'HeadYaw'], [pitch, yaw], 0.1)
    
def set_head(session, pitch, yaw):
    motion = session.service('ALMotion')
    motion.setAngles(['HeadPitch', 'HeadYaw'], [pitch, yaw], 0.1)
    
def tablet_web(session,webpage):
    tabletService = session.service("ALTabletService")
    tabletService.showWebview(webpage)
    
def tablet_close(session):
    tabletService = session.service("ALTabletService")
    tabletService.showWebview()

def connect_session(args):
    s = qi.Session()
    try:
        s.connect("tcp://" + args.ip + ":" + str(args.port))
    except RuntimeError:
        print(
            "Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(
                args.port) + ".\n"
                             "Please check your script arguments. Run with -h option for help.")
        sys.exit(1)
    return s


def send_image(c, vidS, vidC, img):
    try:
        if img:
            c.sendall(img)
        else:
            print("ERR: no data to send.")
            return False
    except KeyboardInterrupt:
        vidS.unsubscribe(vidC)
        c.close()
        print("Process interrupted, cancelling process.")
        return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. Default 127.0.0.1")
    parser.add_argument("--port", type=int, default=9559,
                        help="Pepper port number. Default 9559.")
    parser.add_argument("--send_port", type=int, default=12345,
                        help="Pepper port number. Default 12345.")
    parser.add_argument("--cam_id", type=int, default=3,
                        help="Camera id according to pepper docs. Use 3 for "
                             "stereo camera and 0. Default is 3.")
    parser.add_argument("--fps", type=int, default=15,
                        help="Specify fps with respect to pepper's docs. "
                             "Default is 15.")
    parser.add_argument("--color_id", type=int, default=9,
                        help="Color space id as specified by pepper's docs. "
                             "Default is 9.")
    parser.add_argument("--res_id", type=int, default=14,
                        help="Resolution id as specified by pepper's docs. "
                             "Default is 14.")
    parser.add_argument("--freeze", action='store_true', help="Freeze Pepper's head.")
    parser.add_argument("--debug", action='store_true', help="Debugging info.")

    args = parser.parse_args()

    session = connect_session(args)

    # Freeze Pepper's body
    if args.freeze:
        freeze(session)
    else:
        alive(session)

    sock = create_socket(args)
    conn, addr = sock.accept()  # Connect with client
    video_service, video_client = create_videoService(session, args)

    tts = create_speakService(session)
    awereness = create_trackService(session)
    img = None

    try:
        while True:
            # naoImage = video_service.getDirectRawImageRemote(video_client)
            naoImage = video_service.getImageRemote(video_client)

            # video_service.releaseImage(video_client)

            if naoImage is not None:
                img = bytes(naoImage[6])
                if args.debug:
                    print(naoImage[0], naoImage[1], naoImage[2])
                    print("Size of package", sys.getsizeof(img))

            message = conn.recv(1024)
            if args.debug: print(message)

            if message == 'getImg':
                while True:
                    if send_image(conn, video_service, video_client, img):
                        break
            elif message[:3] == 'say':
                tts.say(message[3:])
            elif message[:5] == 'track':
                awereness.setEngagementMode("FullyEngaged")
                awereness.setEnabled(message[6:10] == 'True')
            elif message[:3] == 'nod':
                nod(session)
            elif message[:4] == 'head':
                print("This is a test")
                move_head(session, float(message[5:9]), float(message[10:14]))
            elif message[:4] == 'shea':
                print("This is a test")
                move_head(session, float(message[5:9]), float(message[10:14]))
            elif message[:4] == 'idle':
                idle(session)
            elif message[:6] == 'tablet':
                tablet_web(session,message[6:])
            elif message[:12] == 'tablet_close':
                tablet_close(session)
            elif message[:9] == 'tabletimg':
                tablet_web(session,message[9:])
            else:
                print("ERR: invalid key..")
                break
    except:
        # Close connections
        print("Closing connections...")
        video_service.unsubscribe(video_client)
        sock.close()
