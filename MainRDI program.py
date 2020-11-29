
import RPi.GPIO as GPIO
import time
import socket

def setup(): ### Set GPIO settings as well as designed voltage/Needs from GPI
    GPIO.setmode(GPIO.BOARD)       # use PHYSICAL GPIO Numbering
    GPIO.setup(ledPin, GPIO.OUT)   # set the ledPin to OUTPUT mode
    GPIO.output(ledPin, GPIO.low)  # make ledPin output LOW level
    print ('using pin%d'%ledPin)

def hardwarefunction(data):
    while True:
        GPIO.output(ledPin, GPIO.HIGH)  # make ledPin output HIGH level to turn on led
        print ('led turned on >>>')     # print information on terminal
        time.sleep(3)                   # Wait for 1 second
        GPIO.output(ledPin, GPIO.LOW)   # make ledPin output LOW level to turn off led
        #print ('led turned off <<<')
        #time.sleep(1)                   # Wait for 1 second

def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0]


def setupconnection():
    HOST =  get_ip_address()  # The server's hostname or IP address
    PORT = 12345        # The port used by the server

    hostname = socket.gethostname()    
    IPAddr = get_ip_address()

    print("Your Computer Name is:" + hostname)    
    print("Your Computer IP Address is:" + IPAddr)
    connected = False


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((IPAddr, PORT))

        print("connection lost... reconnecting")
        connected = False

        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                data.decode()
                print("RECEIVED DATA:" + " TYPE" + str(type(data)) + " Value:" + str(data)
                hardwarefunction(data))
                try:
                loop()

                except KeyboardInterrupt:   # Press ctrl-c to end the program.
                    destroy()

                conn.sendall(b"+")
                print(conn.fileno())
            else
                print("User Has exited Connection")


def destroy():
    GPIO.cleanup()                      # Release all GPIO

if __name__ == '__main__':    # Program entrance
    print ('Program is starting ... \n')
    ledPin = 11  # define ledPin
    setup()
    setupconnection()


