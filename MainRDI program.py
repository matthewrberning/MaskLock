
import RPi.GPIO as GPIO
import time
import socket

def setup(): ### Set GPIO settings as well as designed voltage/Needs from GPI
    GPIO.setmode(GPIO.BOARD)       # use PHYSICAL GPIO Numbering
    GPIO.setup(ledPin, GPIO.OUT)   # set the ledPin to OUTPUT mode
    GPIO.output(ledPin, GPIO.LOW)  # make ledPin output LOW level
    print ('using pin%d'%ledPin)
def destroy():
    GPIO.cleanup()                      # Release all GPIO

def hardwarefunction(code,waittime):
    if code == '0':
        print("Null Code - No Mask Detected Seng Signal on this code to Send message to user")
        
    
    elif code == '1':
        print("CODE 1")
        GPIO.output(ledPin, GPIO.HIGH)  # make ledPin output HIGH level to turn on led
        print ('led turned on >>>')     # print information on terminal
        time.sleep(int(waittime))                   # Wait for 1 second
        GPIO.output(ledPin, GPIO.LOW)   # make ledPin output LOW level to turn off led
        print ('led turned off <<<')
        #time.sleep(1)                   # Wait for 1 second

    elif code == '2':
       print("CODE 2")
    else:
        print("Code not set up for execution on GPIO")
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

        print("Starting Connection")
        connected = False

        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:

                try:
                   data = conn.recv(1024)
                   data = data.decode()
                   code = data[0]
                   waittime = data[-1]
                   print("RECEIVED DATA:" + " TYPE" + str(type(data)) + " Value:" + str(code) + " wait time:" + waittime)
                   hardwarefunction(code,waittime)
                   conn.sendall(bytes([1]))
                   print(conn.fileno())
                except ConnectionResetError:
                    print("Connection Lost Listening Again")
                    
                    s.listen()
                    conn, addr = s.accept()



            
            

if __name__ == '__main__':    # Program entrance
    print ('Program is starting ... \n')
    ledPin = 11  # define ledPin
    setup()
    try:
        setupconnection()
    except KeyboardInterrupt:   # Press ctrl-c to end the program.
        print("Shutting Down Connection")
        destroy()

           
